package main

import (
	"fmt"
	"math"
	"time"
)

func main() {
	const nt int = 100
	const nx int = 100
	const ny int = 100
	var dt float64 = 0.001
	var rho float64 = 1.0
	var nu float64 = 0.1
	var dx float64 = 2 / (float64(nx) - 1)
	var dy float64 = 2 / (float64(ny) - 1)
	var u = zeros(nx, ny)
	var v = zeros(nx, ny)
	var p = zeros(nx, ny)
	
	start := time.Now()
	cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)
	end := time.Now()
	fmt.Println(end.Sub(start))
}

func zeros(rows int, cols int) [][]float64 {
	a := make([][]float64, rows)
	for i := range a {
		a[i] = make([]float64, cols)
	}

	return a
}

func build_up_b(b [][]float64, rho float64, dt float64, u [][]float64, v [][]float64, dx float64, dy float64) [][]float64 {

	var rows = len(b)
	var cols = len(b[0])

	for i := 1; i < (rows - 1); i++ {
		for j := 1; j < (cols - 1); j++ {
			b[i][j] = (rho * (1/dt*
				((u[i][j+1]-u[i][j-1])/
					(2*dx)+(v[i+1][j]-v[i-1][j])/(2*dy)) -
				math.Pow(((u[i][j+1]-u[i][j-1])/(2*dx)), 2) -
				2*((u[i+1][j]-u[i-1][j])/(2*dy)*
					(v[i][j+1]-v[i][j-1])/(2*dx)) -
				math.Pow(((v[i+1][j]-v[i-1][j])/(2*dy)), 2)))
		}
	}

	return b
}

func copy(src [][]float64, dest [][]float64) {

	for i := range src {
		for j := range src[i] {
			dest[i][j] = src[i][j]
		}
	}
}

func copy_with_zeros(src [][]float64, dest [][]float64) {

	for i := range src {
		for j := range src[i] {
			dest[i][j] = 0
		}
	}
}

func pressure_poisson(p [][]float64, dx float64, dy float64, b [][]float64, niter int) [][]float64 {

	var rows = len(p)
	var cols = len(p[0])
	var pn = zeros(rows, cols)

	for iter := 0; iter < niter; iter++ {
		copy(p, pn)
		for i := 1; i < (rows - 1); i++ {
			for j := 1; j < (cols - 1); j++ {

				p[i][j] = (((pn[i][j+1]+pn[i][j-1])*math.Pow(dy, 2)+
					(pn[i+1][j]+pn[i-1][j])*math.Pow(dx, 2))/
					(2*(math.Pow(dx, 2)+math.Pow(dy, 2))) -
					math.Pow(dx, 2)*math.Pow(dy, 2)/(2*(math.Pow(dx, 2)+math.Pow(dy, 2)))*
						b[i][j])
			}
		}

		// python p[:,-1] = p[:,-2]
		// python p[:, 0] = p[:. 1]
		for x := 0; x < rows; x++ {
			p[x][cols-1] = p[x][cols-2]
			p[x][0] = p[x][1]
		}

		// python p[ 0,:] = p[1,:]
		// python p[-1,:] = 0
		for x := 0; x < cols; x++ {
			p[0][x] = p[1][x]
			p[rows-1][x] = 0
		}
	}

	return p
}

func cavity_flow(nt int, u [][]float64, v [][]float64, dt float64, dx float64, dy float64, p [][]float64, rho float64, nu float64) ([][]float64, [][]float64, [][]float64) {

	var rows = len(u)
	var cols = len(u[0])

	var un = zeros(rows, cols)
	var vn = zeros(rows, cols)

	var b = zeros(rows, cols)

	for iter := 0; iter < nt; iter++ {
		
		copy(u, un)
		copy(v, vn)
		b = build_up_b(b, rho, dt, u, v, dx, dy)
		p = pressure_poisson(p, dx, dy, b, 50)

		for i := 1; i < (rows - 1); i++ {
			for j := 1; j < (cols - 1); j++ {

				u[i][j] = (un[i][j] -
					un[i][j]*dt/dx*
						(un[i][j]-un[i][j-1]) -
					vn[i][j]*dt/dy*
						(un[i][j]-un[i-1][j]) -
					dt/(2*rho*dx)*(p[i][j+1]-p[i][j-1]) +
					nu*(dt/math.Pow(dx, 2)*
						(un[i][j+1]-2*un[i][j]+un[i][j-1])+
						dt/math.Pow(dy, 2)*
							(un[i+1][j]-2*un[i][j]+un[i-1][j])))

				v[i][j] = (vn[i][j] -
					un[i][j]*dt/dx*
						(vn[i][j]-vn[i][j-1]) -
					vn[i][j]*dt/dy*
						(vn[i][j]-vn[i-1][j]) -
					dt/(2*rho*dy)*(p[i+1][j]-p[i-1][j]) +
					nu*(dt/math.Pow(dx, 2)*
						(vn[i][j+1]-2*vn[i][j]+vn[i][j-1])+
						dt/math.Pow(dy, 2)*
							(vn[i+1][j]-2*vn[i][j]+vn[i-1][j])))

			}
		}

		// python p[:,-1] = p[:,-2]
		// python p[:, 0] = p[:. 1]
		for x := 0; x < rows; x++ {
			u[x][cols-1] = 0
			u[x][0] = 0
			v[x][cols-1] = 0
			v[x][0] = 0
		}

		// python p[ 0,:] = p[1,:]
		// python p[-1,:] = 0
		for x := 0; x < cols; x++ {
			u[0][x] = 0
			u[rows-1][x] = 1
			v[0][x] = 0
			v[rows-1][x] = 0
		}
	}
	
	return u, v, p
}
