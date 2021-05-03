# lr

Linear Regression From Scratch in Rust

## Dependencies

- [ndarray](https://docs.rs/ndarray/0.15.1/ndarray/)

## Example

```rust
// Construct a linear regression on the data with 1000 iterations and
// the learning rate of 0.00002
let mut lr = LinearRegression::new(features, labels, 1_000, 2e-5);

// Fit the model
lr.fit();

// Make predictions
let preds = lr.predict(&data);
```

## References

- [ndarray](https://docs.rs/ndarray/0.15.1/ndarray/)
- [Linear regression](https://en.wikipedia.org/wiki/Linear_regression)

## License

[MIT License](LICENSE)
