package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

public class Cluster extends ArrayList<double[]> {
	private static final long serialVersionUID = 1L;
	
	/**
	 * The column dimension for the cluster
	 */
	private int n = -1;

	public Cluster() {
		super();
	}
	
	public Cluster copy() {
		final Cluster copy = new Cluster();
		
		if(!this.isEmpty()) {
			final int n = copy.get(0).length;
			for(double[] d: this) {
				double[] copy_array = new double[n];
				System.arraycopy(d, 0, copy_array, 0, n);
				copy.add(copy_array);
			}
		}
		
		return copy;
	}
	
	final private void checkRecord(double[] record) {
		if(n == -1 || isEmpty())
			n = record.length;
		else if(n != record.length)
			throw new DimensionMismatchException(n, record.length);
	}
	
	@Override
	public boolean add(double[] record) {
		checkRecord(record);
		return super.add(record);
	}
	
	@Override
	public void add(int i, double[] record) {
		checkRecord(record);
		super.add(i, record);
	}
	
	@Override
	public boolean addAll(Collection<? extends double[]> d) {
		for(double[] a: d)
			checkRecord(a);
		return super.addAll(d);
	}
	
	@Override
	public boolean addAll(int i, Collection<? extends double[]> d) {
		for(double[] a: d)
			checkRecord(a);
		return super.addAll(i, d);
	}
	
	public double[] centroid() {
		if(n == -1)
			throw new IllegalStateException();
		
		final int m = size();
		if(m == 1) // Avoid poor double precision
			return get(0);
		
		final double[] centroid = new double[n];
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				centroid[j] += get(i)[j];
				
				if(i == size() - 1) // Hack to avoid one more time through N...
					centroid[j] /= (double)m;
			}
		}
		
		return centroid;
	}
	
	@Override
	public double[] set(int i, double[] record) {
		checkRecord(record);
		return super.set(i, record);
	}
	
	/**
	 * Generates a COPY of the data inside the cluster
	 * @param target
	 * @return
	 */
	public double[][] to2DArray() {
		final double[][] copy = new double[this.size()][];
		if(copy.length > 0) {
			final int n = get(0).length;
			int i = 0;
			for(double[] d: this) {
				final double[] row = new double[n];
				System.arraycopy(d, 0, row, 0, n);
				copy[i++] = row;
			}
		}
		
		return copy;
	}
	
	public AbstractRealMatrix toRealMatrix() {
		return new Array2DRowRealMatrix(to2DArray(), false);
	}
}
