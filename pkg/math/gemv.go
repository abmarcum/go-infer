package math

import (
	"go-inference/pkg/quant"
	"runtime"
	"sync"
)

// GEMVEngine manages parallel Matrix-Vector multiplication across CPU cores.
type GEMVEngine struct {
	NumThreads int
	wg         sync.WaitGroup
}

// NewGEMVEngine creates a GEMVEngine with specified thread count.
// If numThreads <= 0, runtime.NumCPU() is used.
func NewGEMVEngine(numThreads int) *GEMVEngine {
	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
	}
	return &GEMVEngine{
		NumThreads: numThreads,
	}
}

// MatMulF32 computes y = W * x where W is rows x cols in float32.
func (e *GEMVEngine) MatMulF32(y, x, w []float32, rows, cols int) {
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowW := w[r*cols : (r+1)*cols]
			y[r] = quant.DotVecF32(x, rowW)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowW := w[r*cols : (r+1)*cols]
				y[r] = quant.DotVecF32(x, rowW)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulQ4_0 computes y = W * x where W is quantized in Q4_0 format.
func (e *GEMVEngine) MatMulQ4_0(y, x []float32, rawQ4 []byte, rows, cols int) {
	bytesPerRow := (cols / 32) * 18
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawQ4[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecQ4_0(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawQ4[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecQ4_0(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulQ8_0 computes y = W * x where W is quantized in Q8_0 format.
func (e *GEMVEngine) MatMulQ8_0(y, x []float32, rawQ8 []byte, rows, cols int) {
	bytesPerRow := (cols / 32) * 34
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawQ8[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecQ8_0(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawQ8[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecQ8_0(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulQ4_K computes y = W * x where W is quantized in Q4_K format.
func (e *GEMVEngine) MatMulQ4_K(y, x []float32, rawQ4K []byte, rows, cols int) {
	bytesPerRow := (cols / 256) * 144
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawQ4K[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecQ4_K(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawQ4K[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecQ4_K(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulQ6_K computes y = W * x where W is quantized in Q6_K format.
func (e *GEMVEngine) MatMulQ6_K(y, x []float32, rawQ6K []byte, rows, cols int) {
	bytesPerRow := (cols / 256) * 210
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawQ6K[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecQ6_K(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawQ6K[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecQ6_K(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulF16 computes y = W * x where W is FP16.
func (e *GEMVEngine) MatMulF16(y, x []float32, rawF16 []byte, rows, cols int) {
	bytesPerRow := cols * 2
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawF16[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecF16(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawF16[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecF16(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulBF16 computes y = W * x where W is BF16.
func (e *GEMVEngine) MatMulBF16(y, x []float32, rawBF16 []byte, rows, cols int) {
	bytesPerRow := cols * 2
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawBF16[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecBF16(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawBF16[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecBF16(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulQ2_K computes y = W * x where W is Q2_K format.
func (e *GEMVEngine) MatMulQ2_K(y, x []float32, rawQ2K []byte, rows, cols int) {
	bytesPerRow := (cols / 256) * 84
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawQ2K[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecQ2_K(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawQ2K[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecQ2_K(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}

// MatMulQ3_K computes y = W * x where W is Q3_K format.
func (e *GEMVEngine) MatMulQ3_K(y, x []float32, rawQ3K []byte, rows, cols int) {
	bytesPerRow := (cols / 256) * 110
	numWorkers := e.NumThreads
	if rows < numWorkers {
		numWorkers = rows
	}
	if numWorkers <= 1 {
		for r := 0; r < rows; r++ {
			rowBytes := rawQ3K[r*bytesPerRow : (r+1)*bytesPerRow]
			y[r] = quant.DotVecQ3_K(x, rowBytes, cols)
		}
		return
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	e.wg.Add(numWorkers)

	for c := 0; c < numWorkers; c++ {
		startRow := c * chunkSize
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}

		go func(start, end int) {
			defer e.wg.Done()
			for r := start; r < end; r++ {
				rowBytes := rawQ3K[r*bytesPerRow : (r+1)*bytesPerRow]
				y[r] = quant.DotVecQ3_K(x, rowBytes, cols)
			}
		}(startRow, endRow)
	}
	e.wg.Wait()
}
