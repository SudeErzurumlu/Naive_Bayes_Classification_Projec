package main

import (
    "fmt"
)

type NaiveBayesClassifier struct {
    classes          []string
    classProb        map[string]float64
    featureProb      map[string]map[int]map[int]float64
}

func (nb *NaiveBayesClassifier) Fit(X [][]int, y []string) {
    totalSamples := len(y)
    nb.classProb = make(map[string]float64)
    nb.featureProb = make(map[string]map[int]map[int]float64)

    for _, cls := range y {
        nb.classProb[cls]++
    }
    for cls := range nb.classProb {
        nb.classProb[cls] /= float64(totalSamples)
        nb.featureProb[cls] = make(map[int]map[int]float64)
    }

    for i, cls := range y {
        for j, feature := range X[i] {
            if nb.featureProb[cls][j] == nil {
                nb.featureProb[cls][j] = make(map[int]float64)
            }
            nb.featureProb[cls][j][feature]++
        }
    }
    for cls := range nb.featureProb {
        for j := range nb.featureProb[cls] {
            total := float64(totalSamples)
            for feature := range nb.featureProb[cls][j] {
                nb.featureProb[cls][j][feature] /= total
            }
        }
    }
}

func main() {
    X := [][]int{{1, 0}, {1, 1}, {0, 1}, {0, 0}}
    y := []string{"Spam", "Spam", "Ham", "Ham"}
    classifier := NaiveBayesClassifier{}
    classifier.Fit(X, y)
    fmt.Println("Naive Bayes Classifier in Go")
}
