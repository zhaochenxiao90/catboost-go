package main

import (
	"log"

	"catboost-go"
)

func main() {
	log.Println("Invoking c library...")

	model, err := catboost.LoadFullModelFromFile("iris.bin")
	if err != nil {
		log.Fatal(err)
	}

	predictions, err := model.CalcModelPrediction([][]float32{{6.7, 2.5, 5.8, 1.8}}, 4, nil, 0)
	if err != nil {
		log.Println(err)
	}
	log.Println(predictions)

	log.Println("Done ")
}
