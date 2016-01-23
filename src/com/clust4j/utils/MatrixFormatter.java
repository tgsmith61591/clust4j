package com.clust4j.utils;

import java.text.NumberFormat;
import java.util.ArrayList;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import static com.clust4j.utils.TableFormatter.ColumnAlignment.RIGHT;

public class MatrixFormatter extends TableFormatter {
	
    public MatrixFormatter() {
    	this(DEFAULT_NUMBER_FORMAT);
    }
    
    public MatrixFormatter(final NumberFormat format) {
    	this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_ROW_PREFIX, DEFAULT_ROW_SUFFIX, 
    		DEFAULT_ROW_SEPARATOR, DEFAULT_COLUMN_SEPARATOR, DEFAULT_WHITE_SPACE, format);
    }
    
    public MatrixFormatter(final String pref, final String suff,
    				 final String rowPref, final String rowSuff,
    				 final String rowSep, final String colSep, final int whiteSpace) {
    	this(pref, suff, rowPref, rowSuff, rowSep, colSep, whiteSpace, DEFAULT_NUMBER_FORMAT);
    }
    
    public MatrixFormatter(final String pref, final String suff,
					 final String rowPref, final String rowSuff,
					 final String rowSep, final String colSep,
					 final int whiteSpace, final NumberFormat format) {
    	super(pref, suff, rowPref, rowSuff, rowSep, colSep, whiteSpace, format);
    }
    
    public String format(double[][] mat) {
    	return format(new Array2DRowRealMatrix(mat, false));
    }
    
    public String format(double[][] mat, int numRows) {
    	return format(new Array2DRowRealMatrix(mat, false), numRows);
    }
    
    public String format(AbstractRealMatrix matrix) {
    	return format(matrix, matrix.getRowDimension());
    }
    
    public String format(AbstractRealMatrix matrix, int numRows) {
    	if(numRows < 1)
    		throw new IllegalArgumentException("numrows must exceed 0");
    	else if(numRows > matrix.getRowDimension())
    		numRows = matrix.getRowDimension();
    	
    	StringBuilder output = new StringBuilder();
    	output.append(prefix+lineSep);
    	
    	final double[][] data = matrix.getData();
    	final int rows = matrix.getRowDimension();
    	final int cols = matrix.getColumnDimension();
    	
    	/* While finding width, go ahead and format */
    	final String[][] formatted = new String[rows][cols];
    	
    	
    	// Need to get the max width for each column
    	ArrayList<Integer> idxToWidth = new ArrayList<Integer>(cols);
    	for(int col = 0; col < cols; col++) {
    		int maxWidth = Integer.MIN_VALUE;
    		for(int row = 0; row < rows; row++) {
    			String f = formatNumber(data[row][col]); //format.format(data[row][col]);
    			int len = f.length();
    			if(len > maxWidth)
    				maxWidth = len;
    			formatted[row][col] = f;
    		}
    		idxToWidth.add(maxWidth);
    	}
    	
    	
    	// Now append plus width, etc.
    	boolean rightJustify = align.equals(RIGHT);
    	for(int row = 0; row < rows; row++) {
    		StringBuilder rowBuild = new StringBuilder();
    		rowBuild.append(rowPrefix);
    		
    		for(int col = 0; col < cols; col++) {
    			StringBuilder entry = new StringBuilder();
    			String f = formatted[row][col];
    			int len = f.length();
    			int colMaxLen = idxToWidth.get(col);
    			int deficit = colMaxLen - len;
    			String appender = getAppenderOfLen(deficit);
    			
    			if(rightJustify) {
    				entry.append(appender);
    				entry.append(f);
    			} else {
    				entry.append(f);
    				entry.append(appender);
    			}
    			
    			rowBuild.append(entry.toString() + (col == cols-1 ? rowSuffix + lineSep : colSepStr));
    		}
    		
    		output.append(rowBuild);
    	}
    	
    	output.append(suffix);
    	return output.toString();
    }
}