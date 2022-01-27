package catboost

// Regression is wrapper over model object that add methods for catboost regression
type Regression struct {
	Model *Model
}

func LoadRegressionFromFile(filename string) (*Regression, error) {
	model, err := LoadFullModelFromFile(filename)
	if err != nil {
		return nil, err
	}
	return &Regression{Model: model}, nil
}

func (r *Regression) PredictRegression(floats [][]float32, floatLength int, cats [][]string, catLength int) ([]float64, error) {
	results, err := r.Model.CalcModelPrediction(floats, floatLength, cats, catLength)
	if err != nil {
		return nil, err
	}
	return results, nil
}

// Close deletes model handler
func (r *Regression) Close() {
	r.Model.Close()
}
