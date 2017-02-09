package com.github.quinngroup;

import org.apache.commons.cli.*;

import org.apache.commons.collections.CollectionUtils;
import org.apache.flink.api.common.functions.*;
import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.aggregation.Aggregations;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.utils.DataSetUtils;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.util.Collector;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.Math.round;

/**
 * Flictionary Learning: Rank-1 Dictionary Learning in Flink!
 * (Java version)
 */
public class R1DL {

	private static Path temporaryPath;

	//
	//	Program
	//
	public static void main(String[] args) throws Exception {

		// Command line options
		Options options = new Options();
		setOptions(options);

		CommandLineParser parser = new DefaultParser();
		final HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd;

		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			System.out.println(e.getMessage());
			showHelp(options, formatter);

			System.exit(1);
			return;
		}

		// show help if option is specified
		if (cmd.hasOption("help")) {
			showHelp(options, formatter);
			return;
		}

		final Integer T = ((Long) cmd.getParsedOptionValue("rows")).intValue();
		final Integer P = ((Long) cmd.getParsedOptionValue("cols")).intValue();

		// convergence stopping criterion
		final Double epsilon = (Double) cmd.getParsedOptionValue("epsilon");

		// dimensionality of the learned dictionary
		final Integer M = ((Long) cmd.getParsedOptionValue("mdicatom")).intValue();

		// enforces sparsity
		final Double nonzero = (Double) cmd.getParsedOptionValue("pnonzero");
		final Long R = round(nonzero * P);

		final Integer maxIterations = P * 10;

		// output file paths (has to have URI prefixes)
		final File file_D = new File(
				new File(cmd.getOptionValue("dictionary")),
				cmd.getOptionValue("prefix") + "_D.txt"
		);
		final File file_z = new File(
				new File(cmd.getOptionValue("output")),
				cmd.getOptionValue("prefix") + "_z.txt"
		);
		// Get a random directory in the temporary path
		// for storing intermediate S matrices.
		temporaryPath = Paths.get(
				cmd.getOptionValue("temporary"),
				"r1dl_data_" + (new Random()).nextInt()
		);

		// Random generator we can reuse. If a seed is specified, use it.
		final Random generator = new Random();
		final Long seed = (Long) cmd.getParsedOptionValue("seed");
		if (seed != null) {
			generator.setSeed(seed);
		}

		// Initialize the Flink environment.
		final ExecutionEnvironment firstEnv = ExecutionEnvironment.getExecutionEnvironment();

		// Read in the input data.
		DataSet<String> rawData = firstEnv.readTextFile(cmd.getOptionValue("input"));
		DataSet<Tuple2<Long, String>> rawS = DataSetUtils.zipWithIndex(rawData);

		// Process S into a DataSet of tuples in this format:
		// (row index, column/vector index, value)
		// (0, (1, 2, 3)) => (0, 0, 1), (0, 1, 2), (0, 2, 3)
		DataSet<Tuple3<Long, Long, Double>> originalS = rawS.flatMap(
				new FlatMapFunction<Tuple2<Long, String>, Tuple3<Long, Long, Double>>() {
					@Override
					public void flatMap(Tuple2<Long, String> raw, Collector<Tuple3<Long, Long, Double>> collector) throws Exception {
						Long id = raw.f0;
						String str = raw.f1;

						String[] tokens = str.split("\t");
						for (Long i = 0L; i < (long) tokens.length; i++) {
							Double val = new Double(tokens[i.intValue()]);
							collector.collect(new Tuple3<>(id, i, val));
						}
					}
				}
		);

		originalS.writeAsCsv(getTemporarySPath(0), FileSystem.WriteMode.OVERWRITE);
		firstEnv.execute("R1DL pre-processing");

		// For each dictionary atom...
		for (int m = 0; m < M; m++) {
			final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
			DataSet<Tuple3<Long, Long, Double>> S = env.readCsvFile(getTemporarySPath(m))
					.types(Long.class, Long.class, Double.class);

			// Generate a random vector, subtract off its mean, and normalize it.
			DataSet<Tuple2<Long, Double>> uOld = env.fromElements(randomVector(T, generator));
			uOld = uOld.reduceGroup(new MeanNormalize());

			// These iterations learn a single atom.
			IterativeDataSet<Tuple2<Long, Double>> uOldIt = uOld.iterate(maxIterations);

			// P2: Vector-matrix multiplication step. Computes v.
			// We don't have numpy or other matrix/vector processing methods, so instead
			// we multiply each value of S by the value of u with the same index as the
			// current row index of S. Then, we add up all the products that share the same
			// vector index to get v. Finally, we select the top R elements by value.
			DataSet<Tuple2<Long, Double>> vTopR = getTopV(R, S, uOldIt);

			// P1: Matrix-vector multiplication step. Computes u.
			DataSet<Tuple3<Long, Long, Double>> uNewPre = vTopR.joinWithHuge(S).where(0).equalTo(1)
					.with(new JoinFunction<Tuple2<Long, Double>, Tuple3<Long, Long, Double>, Tuple3<Long, Long, Double>>() {
						@Override
						public Tuple3<Long, Long, Double> join(Tuple2<Long, Double> vEl, Tuple3<Long, Long, Double> sEl) throws Exception {
							return new Tuple3<>(sEl.f0, sEl.f1, sEl.f2 * vEl.f1);
						}
					})
					.name("MatrixVector");
			DataSet<Tuple2<Long, Double>> uNew = uNewPre.groupBy(0).aggregate(Aggregations.SUM, 2)
					.map(new MapFunction<Tuple3<Long, Long, Double>, Tuple2<Long, Double>>() {
						@Override
						public Tuple2<Long, Double> map(Tuple3<Long, Long, Double> uEl) throws Exception {
							return new Tuple2<>(uEl.f0, uEl.f2);
						}
					});

			// Subtract off the mean and normalize.
			uNew = uNew.reduceGroup(new MeanNormalize());

			// Update for the next iteration.
			// Calculate the difference in the new and old u vectors, and take the magnitude.
			DataSet<Tuple2<Long, Double>> delta = uNew.joinWithHuge(uOldIt).where(0).equalTo(0)
					.with(new JoinFunction<Tuple2<Long, Double>, Tuple2<Long, Double>, Tuple2<Long, Double>>() {
						@Override
						public Tuple2<Long, Double> join(Tuple2<Long, Double> newEl,
						                                 Tuple2<Long, Double> oldEl) throws Exception {
							return new Tuple2<>(newEl.f0, oldEl.f1 - newEl.f1);
						}
					});
			delta = delta.reduceGroup(new GroupReduceFunction<Tuple2<Long, Double>, Tuple2<Long, Double>>() {
				@Override
				public void reduce(Iterable<Tuple2<Long, Double>> iterable, Collector<Tuple2<Long, Double>> collector) throws Exception {
					double mag = 0.0;
					for (Tuple2<Long, Double> t : iterable) {
						mag += Math.pow(t.f1, 2);
					}
					mag = Math.sqrt(mag);
					collector.collect(new Tuple2<>(0L, mag));
				}
			});
			delta = delta.filter(new FilterFunction<Tuple2<Long, Double>>() {
				@Override
				public boolean filter(Tuple2<Long, Double> t) throws Exception {
					return t.f1 > epsilon;
				}
			});

			// Close the iteration. If the magnitude of delta is less than epsilon,
			// convergence has been achieved and the iterations stop.
			DataSet<Tuple2<Long, Double>> uNewFinal = uOldIt.closeWith(uNew, delta);

			uNewFinal.writeAsCsv(file_D.getAbsolutePath() + "." + m,
					FileSystem.WriteMode.OVERWRITE);

			// Find final v and fill in empty spaces in v with zeroes
			DataSet<Tuple2<Long, Double>> vFinal = getTopV(R, S, uNewFinal);
			vFinal = vFinal.union(env.fromElements(generateZeroSequence(P))).groupBy(0)
					.aggregate(Aggregations.SUM, 1);
			vFinal.writeAsCsv(file_z.getAbsolutePath() + "." + m,
					FileSystem.WriteMode.OVERWRITE);

			// P4: Deflation step. Update the primary data matrix S.
			// First, we pair each element of S with the u value that has the same index as the
			// current row index of S.
			DataSet<Tuple4<Long, Long, Double, Double>> tempS = S.join(uNewFinal).where(0).equalTo(0)
					.with(new JoinFunction<Tuple3<Long, Long, Double>,
							Tuple2<Long, Double>, Tuple4<Long, Long, Double, Double>>() {
						@Override
						public Tuple4<Long, Long, Double, Double> join(Tuple3<Long, Long, Double> sEl,
						                                               Tuple2<Long, Double> uEl) throws Exception {
							return new Tuple4<>(sEl.f0, sEl.f1, sEl.f2, uEl.f1);
						}
					});

			// We multiply that u value by each element in v, and store the product within the corresponding
			// S elements that share the same vector position.
			tempS = tempS.joinWithTiny(vFinal).where(1).equalTo(0)
					.with(new JoinFunction<Tuple4<Long, Long, Double, Double>,
							Tuple2<Long, Double>, Tuple4<Long, Long, Double, Double>>() {
						@Override
						public Tuple4<Long, Long, Double, Double> join(Tuple4<Long, Long, Double, Double> sEl,
						                                               Tuple2<Long, Double> vEl) throws Exception {
							return new Tuple4<>(sEl.f0, sEl.f1, sEl.f2, sEl.f3 * vEl.f1);
						}
					});

			// Finally, we subtract this product from the value of S.
			S = tempS.map(new MapFunction<Tuple4<Long, Long, Double, Double>, Tuple3<Long, Long, Double>>() {
				@Override
				public Tuple3<Long, Long, Double> map(Tuple4<Long, Long, Double, Double> sEl) throws Exception {
					return new Tuple3<>(sEl.f0, sEl.f1, sEl.f2 - sEl.f3);
				}
			});

			// Add up all of the differences.
			S = S.groupBy(0, 1).aggregate(Aggregations.SUM, 2);
			S.writeAsCsv(getTemporarySPath(m + 1), FileSystem.WriteMode.OVERWRITE);

			env.execute("R1DL atom #" + m);
		}


	}

	private static void showHelp(Options options, HelpFormatter formatter) {
		formatter.printHelp(".../flink run flink-r1dl.jar",
				"Flictionary Learning: Rank-1 Dictionary Learning in Flink!\n\n",
				options,
				"\nhttps://quinngroup.github.io/",
				true);
	}

	private static String getTemporarySPath(int m) {
		return Paths.get(temporaryPath.toString(), "temp_S." + m).toString();
	}

	/**
	 * Calculate v, and then return the top R elements of v
	 *
	 * @param r    R
	 * @param s    S matrix
	 * @param uOld u
	 * @return Top R elements of v
	 */
	private static DataSet<Tuple2<Long, Double>> getTopV(Long r, DataSet<Tuple3<Long, Long, Double>> s, DataSet<Tuple2<Long, Double>> uOld) {
		DataSet<Tuple2<Long, Double>> v = vectorMatrix(uOld, s);

		return getTopR(r, v);
	}

	/**
	 * Sort a vector and get the top R elements.
	 *
	 * @param r Number of elements to extract.
	 * @param v Vector
	 * @return Truncated, sorted vector
	 */
	private static DataSet<Tuple2<Long, Double>> getTopR(Long r, DataSet<Tuple2<Long, Double>> v) {
		v = sortV(v);

		return v.first(r.intValue());
	}

	/**
	 * Creates a dummy field in a vector (DataSet of (index, value) tuples), uses that field to sort
	 * by value in descending order, then removes the dummy field.
	 *
	 * @param v vector
	 * @return sorted vector
	 */
	private static DataSet<Tuple2<Long, Double>> sortV(DataSet<Tuple2<Long, Double>> v) {
		v = v.map(new MapFunction<Tuple2<Long, Double>, Tuple3<Long, Double, Integer>>() {
			@Override
			public Tuple3<Long, Double, Integer> map(Tuple2<Long, Double> x) throws Exception {
				return new Tuple3<>(x.f0, x.f1, 0);
			}
		}).groupBy(2).sortGroup(1, Order.DESCENDING)
				.reduceGroup(new GroupReduceFunction<Tuple3<Long, Double, Integer>, Tuple2<Long, Double>>() {
					@Override
					public void reduce(Iterable<Tuple3<Long, Double, Integer>> iterable, Collector<Tuple2<Long, Double>> collector) throws Exception {
						for (Tuple3<Long, Double, Integer> t : iterable) {
							collector.collect(new Tuple2<>(t.f0, t.f1));
						}
					}
				});
		return v;
	}


	/**
	 * Generate a vector with a certain number of random elements.
	 * Random numbers generated in the interval [0.0, 1.0).
	 *
	 * @param numElements Number of elements to generate
	 * @return Random vector
	 */
	@SuppressWarnings("unchecked")
	private static Tuple2<Long, Double>[] randomVector(Integer numElements) {
		return randomVector(numElements, null);
	}

	/**
	 * Generate a vector with a certain number of random elements.
	 * Random numbers generated in the interval [0.0, 1.0).
	 *
	 * @param numElements Number of elements to generate
	 * @param generator   Random object to generate numbers (so the seed can be set)
	 * @return Random vector
	 */
	@SuppressWarnings("unchecked")
	private static Tuple2<Long, Double>[] randomVector(Integer numElements, Random generator) {
		Tuple2<Long, Double>[] result = new Tuple2[numElements];
		for (int i = 0; i < numElements; i++) {
			double number;
			if (generator == null) {
				number = Math.random();
			} else {
				number = generator.nextDouble();
			}
			result[i] = new Tuple2<>((long) i, number);
		}
		return result;
	}

	/**
	 * Generate a vector with a certain number of zero elements.
	 *
	 * @param numElements Number of elements
	 * @return Vector of zeroes
	 */
	@SuppressWarnings("unchecked")
	private static Tuple2<Long, Double>[] generateZeroSequence(Integer numElements) {
		Tuple2<Long, Double>[] result = new Tuple2[numElements];
		for (int i = 0; i < numElements; i++) {
			result[i] = new Tuple2<>((long) i, 0.0);
		}
		return result;
	}

	/**
	 * Set command line options.
	 *
	 * @param options Options object which to add options.
	 */
	private static void setOptions(Options options) {
		options.addOption(Option.builder("i")
				.longOpt("input")
				.desc("Input file containing the matrix S.")
				.hasArg()
				.required()
				.build());
		options.addOption(Option.builder("t")
				.longOpt("rows")
				.desc("Number of rows (observations) in the input matrix S.")
				.hasArg()
				.type(Number.class)
				.required()
				.build());
		options.addOption(Option.builder("p")
				.longOpt("cols")
				.desc("Number of columns (features) in the input matrix S.")
				.hasArg()
				.type(Number.class)
				.required()
				.build());
		options.addOption(Option.builder("r")
				.longOpt("pnonzero")
				.desc("Percentage of non-zero elements.")
				.hasArg()
				.type(Number.class)
				.required()
				.build());
		options.addOption(Option.builder("m")
				.longOpt("mdicatom")
				.desc("Number of dictionary atoms.")
				.hasArg()
				.type(Number.class)
				.required()
				.build());
		options.addOption(Option.builder("e")
				.longOpt("epsilon")
				.desc("The value of epsilon.")
				.hasArg()
				.type(Number.class)
				.required()
				.build());
		options.addOption(Option.builder("d")
				.longOpt("dictionary")
				.desc("Output path to dictionary file. (file_D)")
				.hasArg()
				.required()
				.build());
		options.addOption(Option.builder("o")
				.longOpt("output")
				.desc("Output path to z matrix. (file_z)")
				.hasArg()
				.required()
				.build());
		options.addOption(Option.builder("x")
				.longOpt("prefix")
				.desc("Path prefix for output files.")
				.hasArg()
				.required()
				.build());
		options.addOption(Option.builder("y")
				.longOpt("temporary")
				.desc("HDFS or local directory for storing intermediate files.")
				.hasArg()
				.required()
				.build());
		options.addOption(Option.builder("z")
				.longOpt("seed")
				.desc("Random seed. (optional)")
				.hasArg()
				.type(Number.class)
				.build());
		options.addOption(Option.builder("h")
				.longOpt("help")
				.desc("Show this help message.")
				.build());
	}

	/**
	 * P2: Vector-matrix multiplication step. Computes v.
	 *
	 * @param u u
	 * @param S S
	 * @return v
	 */
	private static DataSet<Tuple2<Long, Double>> vectorMatrix(DataSet<Tuple2<Long, Double>> u,
	                                                          DataSet<Tuple3<Long, Long, Double>> S) {
		DataSet<Tuple2<Long, Double>> v = u.joinWithHuge(S).where(0).equalTo(0)
				.with(new JoinFunction<Tuple2<Long, Double>, Tuple3<Long, Long, Double>, Tuple2<Long, Double>>() {
					@Override
					public Tuple2<Long, Double> join(Tuple2<Long, Double> uEl, Tuple3<Long, Long, Double> sEl) throws Exception {
						return new Tuple2<>(sEl.f1, sEl.f2 * uEl.f1);
					}
				}).name("VectorMatrix");

		v = v.groupBy(0).aggregate(Aggregations.SUM, 1);
		return v;
	}

	/**
	 * For each element of a vector, subtract the mean and divide by the magnitude.
	 */
	private static class MeanNormalize implements GroupReduceFunction<Tuple2<Long, Double>, Tuple2<Long, Double>> {
		@Override
		public void reduce(Iterable<Tuple2<Long, Double>> vector, Collector<Tuple2<Long, Double>> collector)
				throws Exception {
			List<Tuple2<Long, Double>> vecList = new ArrayList<>();
			CollectionUtils.addAll(vecList, vector.iterator());

			Double mean = 0.0, mag = 0.0;
			for (Tuple2<Long, Double> item : vecList) {
				mean += item.f1;
				mag += Math.pow(item.f1, 2);
			}

			mean /= vecList.size();
			mag = Math.sqrt(mag);

			for (Tuple2<Long, Double> item : vecList) {
				item.f1 -= mean;
				item.f1 /= mag;
				collector.collect(item);
			}
		}
	}
}
