package catboost

import (
	"testing"
)

func Test_test(t *testing.T) {
	model, err := LoadFullModelFromFile("iris.bin")
	if err != nil {
		t.Fatal(err)
	}
	predictions, err := model.CalcModelPrediction([][]float32{{6.7, 2.5, 5.8, 1.8}}, 4, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(predictions)
}
