naive_bayes_classifier <- function(X, y) {
  classes <- unique(y)
  class_probabilities <- table(y) / length(y)
  feature_probabilities <- list()
  
  # Calculate feature probabilities
  for (cls in classes) {
    cls_indices <- which(y == cls)
    feature_probabilities[[cls]] <- apply(X[cls_indices, ], 2, table) / length(cls_indices)
  }
  
  list(
    predict = function(X_new) {
      predictions <- sapply(1:nrow(X_new), function(i) {
        class_scores <- sapply(classes, function(cls) {
          class_score <- class_probabilities[cls]
          for (j in 1:ncol(X_new)) {
            class_score <- class_score * feature_probabilities[[cls]][[j]][X_new[i, j]]
          }
          class_score
        })
        names(which.max(class_scores))
      })
      predictions
    }
  )
}

# Example usage
X <- matrix(c(1, 0, 1, 1, 0, 1, 0, 0), ncol = 2, byrow = TRUE)
y <- c("Spam", "Spam", "Ham", "Ham")
classifier <- naive_bayes_classifier(X, y)
print("Predictions:")
print(classifier$predict(X))
