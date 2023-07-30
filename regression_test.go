package catboost

import (
	"math"
	"testing"
)

func TestLoadRegressionFromFile(t *testing.T) {
	// see regression.py to know how we get this model
	model, err := LoadRegressionFromFile("regression.bin")
	assertNilError(t, err)
	// this input are the same with inputs in train.py
	t.Run("with eval_data from python script", func(t *testing.T) {
		regression, err := model.PredictRegression(
			[][]float32{{1}, {2}, {2}}, 1,
			[][]string{{"female"}, {"female"}, {"male"}}, 1,
			nil, 0,
			[][][]float32{{{0.2, 0.1, 0.3}, {1.2, 0.3}}, {{0.33, 0.22, 0.4}, {0.98, 0.5}}, {{0.78, 0.29, 0.67}, {0.76, 0.34}}}, []int{3, 2}, 2,
		)
		assertNilError(t, err)
		t.Log(regression)
		assertTrue(t, math.Abs(regression[0]-0.46018641) < 0.00001)
		assertTrue(t, math.Abs(regression[1]-0.47496323) < 0.00001)
		assertTrue(t, math.Abs(regression[2]-0.65977057) < 0.00001)
	})
}
