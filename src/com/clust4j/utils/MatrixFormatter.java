package com.clust4j.utils;

import java.text.NumberFormat;
import java.util.ArrayList;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

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
    	return format(mat, mat.length);
    }
    
    public String format(double[][] mat, int numRows) {
    	MatUtils.checkDimsPermitEmpty(mat);
    	final int rows = mat.length;
    	
    	if(numRows < 1)
    		throw new IllegalArgumentException("numrows must exceed 0");
    	else if(numRows > rows)
    		numRows = rows;
    	
    	StringBuilder output = new StringBuilder();
    	output.append(prefix+lineSep);
    	
    	final double[][] data = MatUtils.copy(mat);
    	
    	// In case of jagged rows, get max col len
    	int max_cols = 0;
    	for(int row = 0; row < numRows; row++)
    		max_cols = FastMath.max(mat[row].length, max_cols);
    	
    	/*// We can allow this for empty arrays...
    	if(max_cols < 1)
    		throw new IllegalArgumentException("max row length is 0");
    	*/
    	
    	
    	/* While finding width, go ahead and format */
    	final String[][] formatted = new String[numRows][max_cols];
    	
    	
    	// Need to get the max width for each column
    	ArrayList<Integer> idxToWidth = new ArrayList<Integer>(max_cols);
    	for(int col = 0; col < max_cols; col++) {
    		int maxWidth = Integer.MIN_VALUE;
    		
    		for(int row = 0; row < numRows; row++) {
    			int thisLen = mat[row].length;
    			
    			String f = col < thisLen ? formatNumber(data[row][col]) : NULL_STR; //format.format(data[row][col]);
    			int len = f.length();
    			if(len > maxWidth)
    				maxWidth = len;
    			
    			formatted[row][col] = f;
    		}
    		
    		idxToWidth.add(maxWidth);
    	}
    	
    	
    	// Now append plus width, etc.
    	boolean rightJustify = align.equals(RIGHT);
    	for(int row = 0; row < numRows; row++) {
    		StringBuilder rowBuild = new StringBuilder();
    		rowBuild.append(rowPrefix);
    		
    		for(int col = 0; col < max_cols; col++) {
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
    			
    			rowBuild.append(entry.toString() + (col == max_cols-1 ? rowSuffix + lineSep : colSepStr));
    		}
    		
    		output.append(rowBuild);
    	}
    	
    	output.append(suffix);
    	return output.toString();
    }
    
    public String format(int[][] mat) {
    	return format(MatUtils.toDouble(mat));
    }
    
    public String format(int[][] mat, int numRows) {
    	return format(MatUtils.toDouble(mat), numRows);
    }
    
    public String format(AbstractRealMatrix matrix) {
    	return format(matrix, matrix.getRowDimension());
    }
    
    public String format(AbstractRealMatrix matrix, int numRows) {
    	return format(matrix.getData(), numRows);
    }
}