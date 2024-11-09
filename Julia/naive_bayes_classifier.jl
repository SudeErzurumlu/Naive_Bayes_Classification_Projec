struct NaiveBayesClassifier
    classes
    class_probabilities
    feature_probabilities
end

function fit!(classifier::NaiveBayesClassifier, X, y)
    classifier.classes, class_counts = unique(y), countmap(y)
    total_samples = length(y)
    
    classifier.class_probabilities = Dict(cls => class_counts[cls] / total_samples for cls in classifier.classes)
    classifier.feature_probabilities = Dict(cls => [countmap(X[i, :]) for i in findall(==(cls), y)] for cls in classifier.classes)
end

function predict(classifier::NaiveBayesClassifier, X)
    [argmax([classifier.class_probabilities[cls] * prod(get.(classifier.feature_probabilities[cls][j], xi, 1e-6) for (j, xi) in enumerate(x)) for cls in classifier.classes]) for x in eachrow(X)]
end

# Example usage
X = [1 0; 1 1; 0 1; 0 0]
y = ["Spam", "Spam", "Ham", "Ham"]
classifier = NaiveBayesClassifier([], Dict(), Dict())
fit!(classifier, X, y)
println("Predictions: ", predict(classifier, X))
