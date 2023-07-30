package catboost

import (
	"fmt"
	"math"
	"runtime"
	"testing"
)

func Test_checkMemory(t *testing.T) {
	t.Skip("this test should be run manually")
	// see regression.py to know how we get this model
	model, err := LoadBinaryClassifierFromFile("full_features.bin")
	if err != nil {
		t.Fatal(err)
	}
	// this input are the same with inputs in train.py
	var run = func(t *testing.T) {
		proba, err := model.PredictProba(
			[][]float32{{1}, {2}, {2}}, 1,
			[][]string{{"female"}, {"female"}, {"male"}}, 1,
			nil, 0,
			[][][]float32{{{0.2, 0.1, 0.3}, {1.2, 0.3}}, {{0.33, 0.22, 0.4}, {0.98, 0.5}}, {{0.78, 0.29, 0.67}, {0.76, 0.34}}}, []int{3, 2}, 2,
		)
		assertNilError(t, err)
		if err != nil {
			t.Fatal(err)
		}
		assertTrue(t, math.Abs(proba[0]-0.43984294) < 0.00001)
		assertTrue(t, math.Abs(proba[1]-0.44554054) < 0.00001)
		assertTrue(t, math.Abs(proba[2]-0.56202416) < 0.00001)
	}
	var n = 10_000_000
	// print memory usage including off-heap memory
	for i := 0; i < n; i++ {
		if i%(n/100) == 0 {
			printMemoryUsage(i)
		}
		run(t)
	}
	printMemoryUsage(n)
	// from given logs we see that memory usage is stable
}

func printMemoryUsage(i int) {
	// force GC to release memory
	runtime.GC()
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	heapAllocKB := memStats.HeapAlloc / 1024
	heapInUseKB := memStats.HeapInuse / 1024
	offHeapKB := (memStats.Sys - memStats.HeapReleased) / 1024
	totalUsedKB := (memStats.HeapInuse + (memStats.Sys - memStats.HeapReleased)) / 1024

	fmt.Printf("Run: %d - HeapAlloc: %d KB, HeapInuse: %d KB, Off-Heap (non-heap) memory: %d KB, Total Used Memory: %d KB\n", i, heapAllocKB, heapInUseKB, offHeapKB, totalUsedKB)
}

func Test_full_features(t *testing.T) {
	// see regression.py to know how we get this model
	model, err := LoadBinaryClassifierFromFile("full_features.bin")
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
		if err != nil {
			t.Fatal(err)
		}
		assertTrue(t, math.Abs(proba[0]-0.43984294) < 0.00001)
		assertTrue(t, math.Abs(proba[1]-0.44554054) < 0.00001)
		assertTrue(t, math.Abs(proba[2]-0.56202416) < 0.00001)
	})
}

func Test_numerical_only(t *testing.T) {
	// see regression.py to know how we get this model
	model, err := LoadBinaryClassifierFromFile("numerical_only.bin")
	if err != nil {
		t.Fatal(err)
	}
	// this input are the same with inputs in train.py
	t.Run("with eval_data from python script", func(t *testing.T) {
		proba, err := model.PredictProba(
			[][]float32{{2, 4, 6}, {3, 5, 7}, {4, 6, 8}}, 3,
			nil, 0,
			nil, 0,
			nil, nil, 0,
		)
		assertNilError(t, err)
		t.Log(proba)
		if err != nil {
			t.Fatal(err)
		}
		assertTrue(t, math.Abs(proba[0]-0.4850642) < 0.00001)
		assertTrue(t, math.Abs(proba[1]-0.55679457) < 0.00001)
		assertTrue(t, math.Abs(proba[2]-0.57136696) < 0.00001)
	})
}

func Test_category_only(t *testing.T) {
	// see regression.py to know how we get this model
	model, err := LoadBinaryClassifierFromFile("category_only.bin")
	if err != nil {
		t.Fatal(err)
	}
	// this input are the same with inputs in train.py
	t.Run("with eval_data from python script", func(t *testing.T) {
		proba, err := model.PredictProba(
			nil, 0,
			[][]string{{"a", "a", "b"}, {"a", "a", "b"}, {"b", "b", "a"}, {"b", "b", "b"}}, 3,
			nil, 0,
			nil, nil, 0,
		)
		assertNilError(t, err)
		t.Log(proba)
		if err != nil {
			t.Fatal(err)
		}
		assertTrue(t, math.Abs(proba[0]-0.61389138) < 0.00001)
		assertTrue(t, math.Abs(proba[1]-0.61389138) < 0.00001)
		assertTrue(t, math.Abs(proba[2]-0.38610862) < 0.00001)
		assertTrue(t, math.Abs(proba[2]-0.38610862) < 0.00001)
	})
}

func Test_embedding_only(t *testing.T) {
	// see regression.py to know how we get this model
	model, err := LoadBinaryClassifierFromFile("embedding_only.bin")
	if err != nil {
		t.Fatal(err)
	}
	// this input are the same with inputs in train.py
	t.Run("with eval_data from python script", func(t *testing.T) {
		proba, err := model.PredictProba(
			nil, 0,
			nil, 0,
			nil, 0,
			[][][]float32{
				{{0.2, 0.1, 0.3}, {1.2, 0.3}},
				{{0.33, 0.22, 0.4}, {0.98, 0.5}},
				{{0.78, 0.29, 0.67}, {0.76, 0.34}},
			}, []int{3, 2}, 2,
		)
		assertNilError(t, err)
		t.Log(proba)
		if err != nil {
			t.Fatal(err)
		}
		assertTrue(t, math.Abs(proba[0]-0.46758147) < 0.00001)
		assertTrue(t, math.Abs(proba[1]-0.46038684) < 0.00001)
		assertTrue(t, math.Abs(proba[2]-0.46038684) < 0.00001)
	})
}

func assertTrue(t *testing.T, v bool) {
	if !v {
		t.Fatal("assert failed")
	}
}

func assertNilError(t *testing.T, err error) {
	if err != nil {
		t.Fatal("assert failed", err)
	}
}
