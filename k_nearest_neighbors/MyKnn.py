valid_metrics = ["euclidean", "manhattan", "minkowski"]
valid_weights = ["distance"]


class MyKnn:
    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = "euclidean",
        p: float = 2.0,
        weights: str | None = None,
    ):
        self.n = n_neighbors

        if metric in valid_metrics:
            self.metric = metric
        elif metric not in valid_metrics:
            print(
                f"Invalid metric: '{metric}'. Use one of the valid options: {valid_metrics}."
            )
            raise Exception

        self.p = p

        if weights and weights in valid_weights:
            self.weights = weights
        elif weights and weights not in valid_weights:
            print(
                f"Invalid weights: '{weights}'. Use one of the valid options: {valid_weights}."
            )
            raise Exception

        self.p = p
