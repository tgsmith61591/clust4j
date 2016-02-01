package com.clust4j.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.GlobalState;
import com.clust4j.log.Loggable;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.VecUtils;

public class DataSet {
	public final static MatrixFormatter DEF_FORMATTER = new MatrixFormatter();
	private final MatrixFormatter formatter;
	
	private Array2DRowRealMatrix data;
	private int[] labels;
	private String[] headers;
	
	
	private static String[] genHeaders(int size) {
		String[] out = new String[size];
		for(int i = 0; i < size; i++)
			out[i] = "V" + i;
		return out;
	}
	
	public DataSet(double[][] data, int[] labels) {
		this(new Array2DRowRealMatrix(data, false /*Going to copy later anyways*/), labels);
	}
	
	public DataSet(Array2DRowRealMatrix data, int[] labels) {
		this(data, labels, genHeaders(data.getColumnDimension()), DEF_FORMATTER, true);
	}
	
	public DataSet(Array2DRowRealMatrix data, int[] labels, MatrixFormatter formatter) {
		this(data, labels, genHeaders(data.getColumnDimension()), formatter, true);
	}
	
	public DataSet(double[][] data, int[] labels, String[] headers) {
		this(new Array2DRowRealMatrix(data, true), labels, headers, DEF_FORMATTER, false);
	}
	
	public DataSet(Array2DRowRealMatrix data, int[] labels, String[] headers) {
		this(data, labels, headers, DEF_FORMATTER);
	}
	
	public DataSet(double[][] data, int[] labels, String[] headers, MatrixFormatter formatter) {
		this(new Array2DRowRealMatrix(data, true), labels, headers, formatter, false);
	}
	
	public DataSet(Array2DRowRealMatrix data, int[] labels, String[] headers, MatrixFormatter formatter) {
		this(data, labels, headers, formatter, true);
	}
	
	public DataSet(Array2DRowRealMatrix data, int[] labels, String[] headers, MatrixFormatter formatter, boolean copyData) {
		// Check to make sure dims match up...
		if(labels.length != data.getRowDimension())
			throw new DimensionMismatchException(labels.length, data.getRowDimension());
		if(headers.length != data.getColumnDimension())
			throw new DimensionMismatchException(headers.length, data.getColumnDimension());
		
		this.data = copyData ? (Array2DRowRealMatrix)data.copy() : data;
		this.labels = VecUtils.copy(labels);
		this.headers = VecUtils.copy(headers);
		this.formatter = null == formatter ? DEF_FORMATTER : formatter;
	}
	
	/**
	 * Return a copy of the data
	 * @return
	 */
	public Array2DRowRealMatrix getData() {
		return (Array2DRowRealMatrix)data.copy();
	}
	
	/**
	 * Return a reference to the data
	 * @return
	 */
	public Array2DRowRealMatrix getDataRef() {
		return data;
	}
	
	/**
	 * Return a copy of the headers
	 * @return
	 */
	public String[] getHeaders() {
		return VecUtils.copy(headers);
	}
	
	/**
	 * Return a reference to the headers
	 * @return
	 */
	public String[] getHeaderRef() {
		return headers;
	}
	
	/**
	 * Return a copy of the labels
	 * @return
	 */
	public int[] getLabels() {
		return VecUtils.copy(labels);
	}
	
	/**
	 * Return a reference to the labels
	 * @return
	 */
	public int[] getLabelRef() {
		return labels;
	}
	
	/**
	 * View the dataset in the log
	 * @param logger
	 */
	public void log(Loggable logger) {
		logger.info(this.toString());
	}
	
	/**
	 * Shuffle the rows (and corresponding labels)
	 * and return the new dataset
	 * in place
	 */
	public DataSet shuffle() {
		return shuffle(GlobalState.DEFAULT_RANDOM_STATE);
	}
	
	/**
	 * Shuffle the rows (and corresponding labels)
	 * and return the new dataset
	 * in place
	 */
	public DataSet shuffle(Random seed) {
		final int m = data.getRowDimension();
		ArrayList<Integer> indices = new ArrayList<Integer>();
		for(int i = 0; i < m; i++)
			indices.add(i);
		
		Collections.shuffle(indices, seed);
		final int[] newLabels = new int[m];
		final double[][] newData = new double[m][];
		
		int j = 0;
		for(Integer idx: indices) {
			newLabels[j] = this.labels[idx];
			newData[j++] = this.data.getRow(idx);
		}
		
		return new DataSet(
			new Array2DRowRealMatrix(newData, false),
			newLabels,
			getHeaders(),
			formatter,
			false
		);
	}
	
	/**
	 * View the dataset in the console
	 */
	public void stdOut() {
		System.out.println(this.toString());
	}

	@Override
	public String toString() {
		return formatter.format(data);
	}
}
