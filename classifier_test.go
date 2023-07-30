package catboost

import (
	"math"
	"testing"
)

func Test_test(t *testing.T) {
	// see regression.py to know how we get this model
	model, err := LoadBinaryClassifierFromFile("classifier.bin")
	if err != nil {
		t.Fatal(err)
	}
	// this input are the same with inputs in train.py
	t.Run("with eval_data from python script", func(t *testing.T) {
		proba, err := model.PredictProba(
			[][]float32{{1}, {2}, {2}}, 1,
			[][]string{{"female"}, {"female"}, {"male"}}, 1,
			nil, 0,
			[][][]float32{{{0.2, 0.1, 0.3}, {1.2, 0.3}}, {{0.33, 0.22, 0.4}, {0.98, 0.5}}, {{0.78, 0.29, 0.67}, {0.76, 0.34}}}, []int{3, 2}, 2,
		)
		assertNilError(t, err)
		t.Log(proba)
		// see the output in train.py
		// # [[0.56015706 0.43984294]
		//#  [0.55445946 0.44554054]
		//#  [0.43797584 0.56202416]]
		if err != nil {
			t.Fatal(err)
		}
		assertTrue(t, math.Abs(proba[0]-0.43984294) < 0.00001)
		assertTrue(t, math.Abs(proba[1]-0.44554054) < 0.00001)
		assertTrue(t, math.Abs(proba[2]-0.56202416) < 0.00001)
	})
}

func assertTrue(t *testing.T, v bool) {
	if !v {
		t.Fatal("assert failed")
	}
}

func assertNilError(t *testing.T, err error) {
	if err != nil {
		t.Fatal("assert failed")
	}
}
