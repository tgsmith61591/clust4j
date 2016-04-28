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
package com.clust4j.utils;

import java.text.NumberFormat;
import java.util.ArrayList;

import org.apache.commons.math3.linear.RealMatrix;

public class MatrixFormatter extends TableFormatter {
	private static final long serialVersionUID = 6065772725783899020L;

	public MatrixFormatter() {
    	this(DEFAULT_NUMBER_FORMAT);
    }
	
	public MatrixFormatter(final ColumnAlignment align) {
		super(align);
	}
    
    public MatrixFormatter(final NumberFormat format) {
    	this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_ROW_PREFIX, DEFAULT_ROW_SUFFIX, 
    		DEFAULT_ROW_SEPARATOR, DEFAULT_COLUMN_SEPARATOR, DEFAULT_WHITE_SPACE, format);
    }
    
    public MatrixFormatter(final String pref, final String suff,
					 final String rowPref, final String rowSuff,
					 final String rowSep, final String colSep,
					 final int whiteSpace, final NumberFormat format) {
    	super(pref, suff, rowPref, rowSuff, rowSep, colSep, whiteSpace, format);
    }
    
    public Table format(double[][] mat) {
    	return format(mat, mat.length);
    }
    
    public Table format(double[][] mat, int numRows) {
    	final ArrayList<Object[]> out = new ArrayList<>();
    	for(double[] d: mat)
    		out.add(doubleToObj(d));
    	return new Table(out, numRows);
    }
    
    public Table format(int[][] mat) {
    	return format(MatUtils.toDouble(mat));
    }
    
    public Table format(int[][] mat, int numRows) {
    	return format(MatUtils.toDouble(mat), numRows);
    }
    
    public Table format(RealMatrix matrix) {
    	return format(matrix, matrix.getRowDimension());
    }
    
    public Table format(RealMatrix matrix, int numRows) {
    	return format(matrix.getData(), numRows);
    }
    
    static Object[] doubleToObj(double[] d) {
    	final Object[] o = new Object[d.length];
    	for(int i = 0; i < o.length; i++)
    		o[i] = (Object)new Double(d[i]);
    	return o;
    }
}