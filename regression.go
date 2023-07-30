package catboost

// Regression is a wrapper over model object that adds methods for catboost regression
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

func (r *Regression) PredictRegression(
	floats [][]float32, floatLength int,
	cats [][]string, catLength int,
	texts [][]string, textLength int,
	embeddings [][][]float32, embeddingDimensions []int, embeddingSize int,
) ([]float64, error) {
	results, err := r.Model.CalcModelPredictionTextAndEmbeddings(
		floats, floatLength,
		cats, catLength,
		texts, textLength,
		embeddings, embeddingDimensions, embeddingSize,
	)
	if err != nil {
		return nil, err
	}
	return results, nil
}

// Close deletes model handler
func (r *Regression) Close() {
	r.Model.Close()
}
