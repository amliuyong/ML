

## Examples
```json
{
  "outputs": [
    "ALL_CATEGORICAL",
    "ALL_NUMERIC"
  ]
}
```
```json
{
  "outputs": [
    "ALL_BINARY",
    "ALL_CATEGORICAL",
    "normalize(ALL_NUMERIC)"
]
}
```

```json
{
  "groups" : {
    "NUMERIC_VARS_QB_20" : "group('sepal_width')",
    "NUMERIC_VARS_QB_10" : "group('petal_length','petal_width','sepal_length')"
  },
  "assignments" : { },
  "outputs" : [ "ALL_CATEGORICAL", "quantile_bin(NUMERIC_VARS_QB_20,20)", "quantile_bin(NUMERIC_VARS_QB_10,10)" ]
}
```


## TEXT

- ngram(textFeature, n)

- osb(textFeature, size)

    "The qick brown fox", {The_quick, The_brown, The_fox}
    "quick brown fox a", {quick_brown, quick_fox, quick_a}

- lowercase(textFeature)

- nopunct(textFeature)

## Numeric

- quantile_bin(numericFeature, n)

- normalize(numericFeature, n)


## Categorical, Text

- cartesian(feature1, feature2)