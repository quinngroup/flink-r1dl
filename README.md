# flink-r1dl
R1DL in Flink using Java. Sister project of [quinngroup/pyflink-r1dl](https://github.com/quinngroup/pyflink-r1dl/).

## Building
```mvn clean install -Pbuild-jar```
You'll get a jar you can use with Flink in the `target/` directory.

## Usage
Run `.../flink run [flink-r1dl jar]` without extra arguments for usage information.

Input files are made up of rows of whitespace-separated numbers (whitespace separating the columns).
