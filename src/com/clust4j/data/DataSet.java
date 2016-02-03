package com.clust4j.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.GlobalState;
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
public class DataSet implements DeepCloneable {
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
		if(null == labels)
			throw new IllegalArgumentException("labels cannot be null");
		if(null == headers)
			throw new IllegalArgumentException("headers cannot be null");
		if(null == data)
			throw new IllegalArgumentException("data cannot be null");
		
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
			
			try { // knowing that this will always throw...
				headers = new String[0];
				data = new Array2DRowRealMatrix(m, 0);
			} catch(NotStrictlyPositiveException nsp) {
				throw new IllegalStateException("cannot drop last column", nsp);
			}
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
			return data.equals(other.data)
				&& headers.equals(other.headers)
				&& labels.equals(other.labels);
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
	
	@Override
	public int hashCode() {
		return 31 
			^ data.hashCode()
			^ headers.hashCode()
			^ labels.hashCode();
	}
	
	private ArrayList<Object[]> buildHead(int length) {
		if(length < 0)
			throw new IllegalArgumentException("numRows cannot be less than 0");
		
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
	
	public void setRow(final int idx, final double[] newRow) {
		final int m = data.getRowDimension();
		if(idx >= m || idx < 0)
			throw new IllegalArgumentException("illegal row index: "+idx);
		
		data.setRow(idx, newRow);
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
			newData[j] = this.data.getRow(idx);
			j++;
		}
		
		return new DataSet(
			new Array2DRowRealMatrix(newData, false),
			newLabels,
			getHeaders(),
			formatter,
			false
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
