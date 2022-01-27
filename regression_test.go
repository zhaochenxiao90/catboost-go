package catboost

import (
	"gotest.tools/assert"
	"math"
	"testing"
)

func TestLoadRegressionFromFile(t *testing.T) {
	// see regression.py to know how we get this model
	model, err := LoadRegressionFromFile("regression.bin")
	assert.NilError(t, err)
	// this input are the same with inputs in regression.py
	t.Run("with eval_data from python script", func(t *testing.T) {
		regression, err := model.PredictRegression(
			[][]float32{{2, 4, 6, 8}, {1, 4, 50, 60}}, 4,
			[][]string{{"a"}, {"b"}}, 2,
		)
		assert.NilError(t, err)
		t.Log(regression)
		assert.Assert(t, math.Abs(regression[0]-15.65772339) < 0.00001)
		assert.Assert(t, math.Abs(regression[1]-20.38869995) < 0.00001)
	})
	t.Run("random data", func(t *testing.T) {
		regression, err := model.PredictRegression(
			[][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}, 4,
			[][]string{{"a"}, {"b"}}, 2,
		)
		assert.NilError(t, err)
		t.Log(regression)
	})
}
