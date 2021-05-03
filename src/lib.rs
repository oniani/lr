use ndarray::prelude::*;

/// A representation of the linear regression
#[derive(Debug)]
pub struct LinearRegression {
    /// Features
    pub features: Array2<f64>,
    /// Labels
    pub labels: Array1<f64>,
    /// Number of iterations
    pub iterations: u64,
    /// Learning rate
    pub learning_rate: f64,
    /// Model weights
    pub weights: Array1<f64>,
    /// Model bias
    pub bias: f64,
}

/// Implement the linear regression
impl LinearRegression {
    /// Create a new model
    fn new(
        features: Array2<f64>,
        labels: Array1<f64>,
        iterations: u64,
        learning_rate: f64,
    ) -> Self {
        LinearRegression {
            features,
            labels,
            iterations,
            learning_rate,
            weights: Array::zeros(0),
            bias: 0.,
        }
    }

    /// Fit the model
    pub fn fit(&mut self) -> () {
        // Number of rows
        let m: usize = self.features.shape()[0];
        // Number of columns
        let n: usize = self.features.shape()[1];

        // Zero-initialize weights
        self.weights = Array::<f64, _>::zeros(n);
        // Zero-initialize bias
        self.bias = 0.;

        // Gradient descent
        for _ in 0..self.iterations {
            // Make a prediction
            let pred = self.features.dot(&self.weights) + self.bias;

            // Calculate gradients
            let diff = &self.labels - pred;
            let dw =
                &self.features.t().dot(&diff).map(|x| (*x) * -2. / m as f64);
            let db = diff.sum() * -2. / m as f64;

            // Update the weights
            self.weights = &self.weights - self.learning_rate * dw;
            self.bias -= self.learning_rate * db;
        }
    }

    /// Make a prediction
    pub fn predict(self, features: &Array2<f64>) -> Array1<f64> {
        features.dot(&self.weights) + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr() {
        // Input
        let xs = Array::<f64, _>::range(0., 100., 1.)
            .into_shape((100, 1))
            .unwrap();
        let ys = Array::<f64, _>::range(0., 100., 1.).map(|x| *x * 2.);

        // Construct a linear regression on the data with 1000 iterations and
        // the learning rate of 0.00002
        let mut lr = LinearRegression::new(xs, ys, 1_000, 2e-5);

        // Fit the model f(x) = 2x
        lr.fit();

        // Dummy data
        let dummy = Array::<f64, _>::range(200., 300., 1.)
            .into_shape((100, 1))
            .unwrap();

        // Make a prediction
        let preds = lr.predict(&dummy).map(|x| (*x).round());

        // Correct output
        let ground_truth = Array::<f64, _>::range(400., 600., 2.);

        // Make sure that the linear regression predicts correctly
        assert_eq!(preds, ground_truth);
    }
}
