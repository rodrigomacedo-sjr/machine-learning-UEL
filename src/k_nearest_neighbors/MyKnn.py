from numpy import sqrt
from pandas.core.frame import DataFrame
from pandas.core.series import Series


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

        self.training_points = []

    def fit(self, data: DataFrame, targets: Series):
        self.training_points.clear()
        for index, row in data.iterrows():
            self.training_points.append([index, row.values, targets[index]])

    def predict(self, data: DataFrame):
        """
        for each point X in data:
           calculate distance to each point Y in training_points
            distance[X,Y] = [dist, class]
            
            sort distance from lowest to highest
            take the first k elements
            return the most common class in those

            in the case of a draw, the values are weighted
        """
        predictions = []
        for index, row in data.iterrows():
            distances = {}
            for point in self.training_points:
                distances[point[0]] = [
                    self.calculate_distance(row.values, point[1]),
                    point[2],
                ]

            count = {}
            sorted_points = sorted(distances.items(), key=lambda item: item[1][0])
            for i in range(self.n):
                if i < len(sorted_points):
                    cls = sorted_points[i][1][1]
                    count[cls] = count.get(cls, 0) + 1

            if count:
                max_count = max(count.values())
                winners = [cls for cls, c in count.items() if c == max_count]
                if len(winners) == 1:
                    predictions.append(winners[0])
                elif self.weights == "distance":
                    weighted_votes = {cls: 0 for cls in winners}
                    for i in range(self.n):
                        if i < len(sorted_points):
                            cls = sorted_points[i][1][1]
                            dist = sorted_points[i][1][0]
                            if cls in weighted_votes:
                                weighted_votes[cls] += 1 / (dist + 1e-8)
                    predictions.append(max(weighted_votes, key=weighted_votes.get))
                else:
                    predictions.append(winners[0])
        return predictions

    def calculate_distance(self, pointA, pointB):
        match self.metric:
            case "euclidean":
                return self.calculate_euclidean_distance(pointA, pointB)
            case "manhattan":
                return self.calculate_manhattan_distance(pointA, pointB)
            case "minkowski":
                return self.calculate_minkowski_distance(pointA, pointB)

    def calculate_euclidean_distance(self, pointA, pointB):
        sum = 0
        for i in range(len(pointA)):
            sum += pow(pointA[i] - pointB[i], 2)
        return sqrt(sum)

    def calculate_manhattan_distance(self, pointA, pointB):
        sum = 0
        for i in range(len(pointA)):
            sum += abs(pointA[i] - pointB[i])
        return sum

    def calculate_minkowski_distance(self, pointA, pointB):
        sum = 0
        for i in range(len(pointA)):
            sum += pow(abs(pointA[i] - pointB[i]), self.p)
        return pow(sum, (1 / self.p))
