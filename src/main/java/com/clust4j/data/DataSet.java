/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.data;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.Clust4j;
import com.clust4j.log.Loggable;
import com.clust4j.utils.DeepCloneable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.TableFormatter;
import com.clust4j.utils.VecUtils;

/**
 * A lightweight dataset wrapper that stores information on
 * header names, matrix data and classification labels.
 * @author Taylor G Smith
 */
public class DataSet extends Clust4j implements DeepCloneable, java.io.Serializable {
	private static final long serialVersionUID = -1203771047711850121L;
	
	final static String COL_PREFIX = "V";
	final static int DEF_HEAD = 6;
	
	public final static TableFormatter TABLE_FORMATTER= new TableFormatter();
	public final static MatrixFormatter DEF_FORMATTER = new MatrixFormatter();
	private final MatrixFormatter formatter;
	
	private Array2DRowRealMatrix data;
	private int[] labels;
	private String[] headers;
	
	
	private static String[] genHeaders(int size) {
		String[] out = new String[size];
		for(int i = 0; i < size; i++)
			out[i] = COL_PREFIX + i;
		return out;
	}
	
	public DataSet(double[][] data) {
		this(new Array2DRowRealMatrix(data, false /*Going to copy later anyways*/));
	}
	
	public DataSet(Array2DRowRealMatrix data) {
		this(data, genHeaders(data.getColumnDimension()));
	}
	
	public DataSet(double[][] data, String[] headers) {
		this(new Array2DRowRealMatrix(data, false /*Going to copy later anyways*/), headers);
	}
	
	public DataSet(Array2DRowRealMatrix data, String[] headers) {
		this(data, null, headers);
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
	
	public DataSet(Array2DRowRealMatrix data, int[] labels, String[] hdrz, MatrixFormatter formatter, boolean copyData) {
		
		/*// we should allow this behavior...
		if(null == labels)
			throw new IllegalArgumentException("labels cannot be null");
		*/

		if(null == data)
			throw new IllegalArgumentException("data cannot be null");
		if(null == hdrz)
			this.headers = genHeaders(data.getColumnDimension());
		else 
			this.headers = VecUtils.copy(hdrz);
		
		
		// Check to make sure dims match up...
		if((null != labels) && labels.length != data.getRowDimension())
			throw new DimensionMismatchException(labels.length, data.getRowDimension());
		if(this.headers.length != data.getColumnDimension())
			throw new DimensionMismatchException(this.headers.length, data.getColumnDimension());
		
		this.data = copyData ? (Array2DRowRealMatrix)data.copy() : data;
		this.labels = VecUtils.copy(labels);
		this.formatter = null == formatter ? DEF_FORMATTER : formatter;
	}
	
	public void addColumn(double[] col) {
		addColumn(COL_PREFIX + numCols(), col);
	}
	
	public void addColumns(double[][] cols) {
		MatUtils.checkDims(cols);
		
		final int n = data.getColumnDimension(), length = n + cols[0].length;
		final String[] newCols = new String[cols[0].length];
		for(int i = n, j = 0; i < length; i++, j++)
			newCols[j] = COL_PREFIX + i;
		
		addColumns(newCols, cols);
	}
	
	public void addColumn(String s, double[] col) {
		VecUtils.checkDims(col);
		
		final int m = col.length;
		if(m != data.getRowDimension())
			throw new DimensionMismatchException(m, data.getRowDimension());
		
		final int n = data.getColumnDimension();
		s = null == s ? (COL_PREFIX + n) : s;
		
		String[] newHeaders = new String[n + 1];
		double[][] newData = new double[m][n + 1];
		double[][] oldData = data.getDataRef();
		
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n + 1; j++) {
				if(i == 0)
					newHeaders[j] = j != n ? headers[j]: s;
				newData[i][j] = j != n ? oldData[i][j] : col[i];
			}
		}
				
		this.headers = newHeaders;
		this.data = new Array2DRowRealMatrix(newData, false);
	}
	
	public void addColumns(String[] s, double[][] cols) {
		MatUtils.checkDims(cols);
		
		final int m = cols.length;
		if(m != data.getRowDimension())
			throw new DimensionMismatchException(m, data.getRowDimension());
		
		int i, j;
		final int n = data.getColumnDimension(), newN = n + cols[0].length;
		
		// build headers
		if(null == s) {
			s = new String[cols[0].length];
			for(i = 0, j = n; i < cols[0].length; i++, j++)
				s[i] = COL_PREFIX + j;
		} else {
			// Ensure no nulls
			for(i = 0, j = n; i < cols[0].length; i++, j++)
				s[i] = null == s[i] ? (COL_PREFIX + j) : s[i];
		}
		
		
		String[] newHeaders = new String[newN];
		double[][] newData = new double[m][newN];
		double[][] oldData = data.getDataRef();
		
		for(i = 0; i < m; i++) {
			for(j = 0; j < newN; j++) {
				if(i == 0) {
					newHeaders[j] = j < n ? headers[j]: s[j - n];
				}
					
				newData[i][j] = j < n ? oldData[i][j] : cols[i][j - n];
			}
		}
				
		this.headers = newHeaders;
		this.data = new Array2DRowRealMatrix(newData, false);
	}
	
	@Override
	public DataSet copy() {
		return new DataSet(data, labels, headers, formatter, true);
	}
	
	public double[] dropCol(String nm) {
		return dropCol(getColumnIdx(nm));
	}
	
	public double[] dropCol(int idx) {
		double[] res;
		if(idx >= numCols() || idx < 0)
			throw new IllegalArgumentException("illegal column index: "+idx);
		
		final int m = numRows(), n = numCols();
		final double[][] dataRef = data.getDataRef();
		
		// We know idx won't throw exception
		res = data.getColumn(idx);
		
		
		if(n == 1) {
			throw new IllegalStateException("cannot drop last column");
		} else {
			double[][] newData = new double[m][n - 1];
			String[] newHeader = new String[n - 1];
			
			for(int i = 0; i < m; i++) {
				int k = 0;
				for(int j = 0; j < n; j++) {
					if(j == idx)
						continue;
					else {
						if(i == 0) // On first iter, also reassign headers
							newHeader[k] = headers[j];
						newData[i][k] = dataRef[i][j];
						k++;
					}
				}
			}
			
			data = new Array2DRowRealMatrix(newData, false);
			headers = newHeader;
		}
		
		return res;
	}
	
	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o instanceof DataSet) {
			DataSet other = (DataSet)o;
			System.out.println(VecUtils.equalsExactly(labels, other.labels));
			
			return MatUtils.equalsExactly(data.getDataRef(), other.data.getDataRef())
				&& VecUtils.equalsExactly(headers, other.headers)
				&& VecUtils.equalsExactly(labels, other.labels);
		}
		
		return false;
	}
	
	/**
	 * Return a copy of the data
	 * @return
	 */
	public Array2DRowRealMatrix getData() {
		return (Array2DRowRealMatrix)data.copy();
	}
	
	/**
	 * Returns the column index of the header. If
	 * multiple columns share the same name (bad practice),
	 * returns the first which meets the criteria.
	 * @param header
	 * @return
	 */
	private int getColumnIdx(String header) {
		int idx = 0;
		boolean found = false;
		for(String head: headers) {
			if(head.equals(header)) {
				found = true;
				break;
			}
			
			idx++;
		}
			
		if(!found)
			throw new IllegalArgumentException("no such header: "+header);
		
		return idx;
	}
	
	/**
	 * Return a copy of the column 
	 * corresponding to the header
	 * @param header
	 * @return
	 */
	public double[] getColumn(String header) {
		return getColumn(getColumnIdx(header));
	}
	
	/**
	 * Return a copy of the column 
	 * corresponding to the header
	 * @param header
	 * @return
	 */
	public double[] getColumn(int i) {
		return data.getColumn(i);
	}
	
	/**
	 * Return a reference to the data
	 * @return
	 */
	public Array2DRowRealMatrix getDataRef() {
		return data;
	}
	
	/**
	 * Get the entry at the given row/col indices
	 * @param row
	 * @param col
	 * @return
	 */
	public double getEntry(int row, int col) {
		return this.data.getEntry(row, col);
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
		return null == labels ? null : VecUtils.copy(labels);
	}
	
	/**
	 * Return a reference to the labels
	 * @return
	 */
	public int[] getLabelRef() {
		return labels;
	}
	
	@Override
	public int hashCode() {
		return 31 
			^ data.hashCode()
			^ headers.hashCode()
			^ labels.hashCode();
	}
	
	private ArrayList<Object[]> buildHead(int length) {
		if(length < 1)
			throw new IllegalArgumentException("length cannot be less than 1");
		
		int n = data.getColumnDimension();
		ArrayList<Object[]> o = new ArrayList<Object[]>();
		double[][] d = data.getDataRef();
		o.add(new Object[n]); // There's always one extra row
		
		for(int i = 0; i < length; i++) {
			o.add(new Object[n]);
			
			for(int j = 0; j < n; j++) {
				if(i == 0) {
					o.get(i)[j] = headers[j];
				}
				
				o.get(i+1)[j] = d[i][j];
			}
		}
		
		return o;
	}
	
	public void head() {
		head(DEF_HEAD);
	}
	
	public void head(int numRows) {
		System.out.println(TABLE_FORMATTER.format(buildHead(numRows)));
	}
	
	/**
	 * View the dataset in the log
	 * @param logger
	 */
	public void log(Loggable logger) {
		logger.info(this.toString());
	}
	
	public int numCols() {
		return data.getColumnDimension();
	}
	
	public int numRows() {
		return data.getRowDimension();
	}
	
	public void setColumn(String name, final double[] col) {
		setColumn(getColumnIdx(name), col);
	}
	
	public void setColumn(final int idx, final double[] col) {
		final int n = data.getColumnDimension();
		if(idx >= n || idx < 0)
			throw new IllegalArgumentException("illegal column index: "+idx);
		
		data.setColumn(idx, col);
	}
	
	/**
	 * Set the indices of row/col to the new value and
	 * return the old value
	 * @param row
	 * @param col
	 * @param newValue
	 * @return
	 */
	public double setEntry(int row, int col, double newValue) {
		double d = getEntry(row, col);
		this.data.setEntry(row, col, newValue);
		return d;
	}
	
	public void setLabels(final int[] labels) {
		if(null == labels) // null out existing labels
			this.labels = labels;
		else if(labels.length == data.getRowDimension()) {
			this.labels = labels;
		} else {
			throw new DimensionMismatchException(labels.length, data.getRowDimension());
		}
	}
	
	public void setRow(final int idx, final double[] newRow) {
		final int m = data.getRowDimension();
		if(idx >= m || idx < 0)
			throw new IllegalArgumentException("illegal row index: "+idx);
		
		data.setRow(idx, newRow);
	}
	
	/**
	 * Shuffle the rows (and corresponding labels, if they exist)
	 * and return the new dataset
	 * in place
	 */
	public DataSet shuffle() {
		final int m = numRows();
		boolean has_labels = null != labels; // if the labels are null, there are no labels to shuffle...
		
		/*
		 * Generate range of indices...
		 */
		ArrayList<Integer> indices = new ArrayList<Integer>();
		for(int i = 0; i < m; i++)
			indices.add(i);
		
		/*
		 * Shuffle indices in place...
		 */
		Collections.shuffle(indices);
		final int[] newLabels = has_labels ? new int[m] : null;
		final double[][] newData = new double[m][];
		
		/*
		 * Reorder things...
		 */
		int j = 0;
		for(Integer idx: indices) {
			if(has_labels) {
				newLabels[j] = this.labels[idx];
			}
			
			newData[j] = VecUtils.copy(this.data.getRow(idx));
			j++;
		}
		
		return new DataSet(
			new Array2DRowRealMatrix(newData, true),
			newLabels,
			getHeaders(),
			formatter,
			false
		);
	}
	
	public DataSet slice(int startInc, int endExc) {
		int[] labs = (null == labels) ? null : VecUtils.slice(labels, startInc, endExc);
		
		return new DataSet(
			MatUtils.slice(data.getDataRef(), startInc, endExc),
			labs,
			getHeaders()
		);
	}
	
	public void sortAscInPlace(String col) {
		sortAscInPlace(getColumnIdx(col));
	}
	
	public void sortAscInPlace(int colIdx) {
		if(colIdx < 0 || colIdx >= data.getColumnDimension())
			throw new IllegalArgumentException("col out of bounds");
		
		double[][] dataRef = data.getDataRef();
		data = new Array2DRowRealMatrix(MatUtils.sortAscByCol(dataRef, colIdx), false);
	}
	
	public void sortDescInPlace(String col) {
		sortDescInPlace(getColumnIdx(col));
	}
	
	public void sortDescInPlace(int colIdx) {
		if(colIdx < 0 || colIdx >= data.getColumnDimension())
			throw new IllegalArgumentException("col out of bounds");

		double[][] dataRef = data.getDataRef();
		data = new Array2DRowRealMatrix(MatUtils.sortDescByCol(dataRef, colIdx), false);
	}
	
	/**
	 * View the dataset in the console
	 */
	public void stdOut() {
		System.out.println(this.toString());
	}
	
	/**
	 * Write the dataset to a CSV
	 * @param header
	 * @throws IOException
	 */
	public void toFlatFile(boolean header, final File file) throws IOException {
		toFlatFile(header, file, ',');
	}
	
	/**
	 * Write the dataset to a flat file
	 * @param header
	 * @param sep
	 * @throws IOException
	 */
	public void toFlatFile(boolean header, final File file, char sep) throws IOException {
		synchronized(this) {
			boolean target = null != labels;
			
			int idx = 0, row_idx = 0;
			Object[] new_row;
			String[] output = new String[this.numRows() + (header?1:0)];
			
			/*
			 * If header, append.
			 */
			if(header) {
				new_row = new Object[this.headers.length + (target?1:0)];
				for(int i = 0; i < this.headers.length; i++) {
					new_row[i] = this.headers[i];
				}
				
				if(target) new_row[new_row.length - 1] = "target";
				output[idx++] = toString(new_row, sep);
			}
			
			/*
			 * Stringify data...
			 */
			for(double[] row: this.data.getData()) {
				new_row = new Object[this.headers.length + (target?1:0)];
				for(int i = 0; i < row.length; i++) {
					new_row[i] = row[i];
				}
				
				if(target) new_row[new_row.length - 1] = this.labels[row_idx++];
				output[idx++] = toString(new_row, sep);
			}
			
			/*
			 * Write the bytes...
			 */
			BufferedWriter bw = null;
			try {
				bw = new BufferedWriter(new FileWriter(file));
				
				String out, newline = System.getProperty("line.separator");
				for(int i = 0; i < output.length; i++) {
					out = output[i];
					bw.write(out);
					if(i!=output.length-1) bw.write(newline);
				}
			} finally {
				try {
					bw.close();
				} catch(IOException e) {
					// ignore...
				}
			}
		}
	}
	
	private static String toString(Object[] obj, char sep) {
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < obj.length; i++) {
			sb.append(obj[i]);
			if(i!=obj.length - 1) sb.append(sep);
		}
		
		return sb.toString();
	}

	@Override
	public String toString() {
		String ls = System.getProperty("line.separator");
		String lsls = ls + ls;
		
		StringBuilder sb = new StringBuilder();
		sb.append("Headers:" + ls);
		sb.append(Arrays.toString(headers) + lsls);
		
		sb.append("Data:");
		sb.append(formatter.format(data) + ls);
		
		sb.append("Labels:"+ls);
		sb.append(Arrays.toString(labels));
		
		return sb.toString();
	}
}
