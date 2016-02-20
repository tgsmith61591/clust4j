package com.clust4j.utils;

import static com.clust4j.utils.TableFormatter.ColumnAlignment.LEFT;
import static com.clust4j.utils.TableFormatter.ColumnAlignment.RIGHT;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

public class TableFormatter {
	public static enum ColumnAlignment {
		LEFT, 
		RIGHT
	}
	
    /** The default prefix: "". */
    protected static final String DEFAULT_PREFIX = "";
    /** The default suffix: "". */
    protected static final String DEFAULT_SUFFIX = "";
    /** The default row prefix: "". */
    protected static final String DEFAULT_ROW_PREFIX = "";
    /** The default row suffix: "". */
    protected static final String DEFAULT_ROW_SUFFIX = "";
    /** The default row separator: "\n". */
    protected static final String DEFAULT_ROW_SEPARATOR = System.getProperty("line.separator");
    /** The default column separator: "". */
    protected static final String DEFAULT_COLUMN_SEPARATOR = "";
    /** The default locale */
    protected static final Locale DEFAULT_LOCALE = Locale.US;
    /** The default number format */
    protected static final NumberFormat DEFAULT_NUMBER_FORMAT = NumberFormat.getInstance(DEFAULT_LOCALE); 
    /** The default whitespace between columns */
    protected static final int DEFAULT_WHITE_SPACE = 4;
    /** Default column alignment */
    protected static final ColumnAlignment DEFAULT_ALIGNMENT = RIGHT;
    static final String NULL_STR = "--";

    
    
    /** Prefix. */
    protected final String prefix;
    /** Suffix. */
    protected final String suffix;
    /** Row prefix. */
    protected final String rowPrefix;
    /** Row suffix. */
    protected final String rowSuffix;
    /** Row separator. */
    protected final String rowSeparator;
    /** Column separator. */
    protected final String columnSeparator;
    /** The format used for components. */
    protected final NumberFormat format;
    /** The whitespace between cols */
    protected final int whiteSpace;
	/** Line sep */
    protected final String lineSep;
	/** Between col string constructed from whitespace and char */
    protected final String colSepStr;
	/** Column alignment for printing */
	protected ColumnAlignment align = DEFAULT_ALIGNMENT;
	
    public TableFormatter() {
    	this(DEFAULT_NUMBER_FORMAT);
    }
    
    public TableFormatter(final NumberFormat format) {
    	this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_ROW_PREFIX, DEFAULT_ROW_SUFFIX, 
    		DEFAULT_ROW_SEPARATOR, DEFAULT_COLUMN_SEPARATOR, DEFAULT_WHITE_SPACE, format);
    }
    
    public TableFormatter(final String pref, final String suff,
    				 final String rowPref, final String rowSuff,
    				 final String rowSep, final String colSep, final int whiteSpace) {
    	this(pref, suff, rowPref, rowSuff, rowSep, colSep, whiteSpace, DEFAULT_NUMBER_FORMAT);
    }
    
    public TableFormatter(final String pref, final String suff,
					 final String rowPref, final String rowSuff,
					 final String rowSep, final String colSep,
					 final int whiteSpace, final NumberFormat format) {
    	this.prefix = pref;
    	this.suffix = suff;
    	this.rowPrefix = rowPref;
    	this.rowSuffix = rowSuff;
    	this.rowSeparator = rowSep;
    	
    	this.columnSeparator = colSep;
    	this.whiteSpace = whiteSpace%2==0 ? whiteSpace : whiteSpace + 1;
    	String ws = getAppenderOfLen(this.whiteSpace/2);
    	this.colSepStr = ws + this.columnSeparator + ws; //"  |  " or the likes
    	
    	this.format = format;
    	this.lineSep = System.getProperty("line.separator");
    }
    
    protected String formatNumber(Object o) {
    	String f;
    	if(isNumeric(o)) {
			Double tmpd = ((Number)o).doubleValue();
			f = tmpd.isNaN() ? "NaN" : 
				tmpd.isInfinite() ? (tmpd.equals(Double.NEGATIVE_INFINITY) ? "-Inf" : "Inf") : 
					format.format(tmpd);
		} else {
			f = o.toString();
		}
    	
    	return f;
    }
    
    public String format(ArrayList<Object[]> rows) {
    	return format(rows, rows.size());
    }
    
    public String format(ArrayList<Object[]> matrix, int numRows) {
    	final int rows = matrix.size();
    	if(numRows < 1)
    		throw new IllegalArgumentException("numrows must exceed 0");
    	else if(numRows > rows)
    		numRows = rows;
    	
    	StringBuilder output = new StringBuilder();
    	output.append(prefix+lineSep);

    	final Object[][] data = new Object[numRows][];
    	for(int i = 0; i < numRows; i++) {
    		Object[] matI = matrix.get(i);
    		data[i] = matI;
    	}
    	
    	// Get the max num columns...
    	int largestSoFar = Integer.MIN_VALUE;
    	for(Object[] oo : data)
    		if(oo.length > largestSoFar)
    			largestSoFar = oo.length;
    	
    	// Assign as max...
    	final int cols = largestSoFar;
    	
    	/* While finding width, go ahead and format */
    	final String[][] formatted = new String[numRows][cols];
    	
    	
    	// Need to get the max width for each column
    	ArrayList<Integer> idxToWidth = new ArrayList<Integer>(cols);
    	for(int col = 0; col < cols; col++) {
    		int maxWidth = Integer.MIN_VALUE;
    		for(int row = 0; row < numRows; row++) {
    			String f;
    			int len;
    			
    			if(data[row].length <= col) {
    				f = ""; // Set to empty, if not exists...
    			} else {
    				f = formatNumber(data[row][col]);
    			}

    			len = f.length();
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
    
    
    
    public ColumnAlignment getAlignment() {
    	return align;
    }
    
    protected static String getAppenderOfLen(int n) {
    	if(n == 0)
    		return "";
    	
    	char[] whiteSpaceArr = new char[n];
    	Arrays.fill(whiteSpaceArr, ' ');
    	return new String(whiteSpaceArr);
    }
    
    public String getPrefix() {
    	return prefix;
    }
    
    public String getSuffix() {
    	return suffix;
    }
    
    public String getRowPrefix() {
    	return rowPrefix;
    }
    
    public String getRowSuffix() {
    	return rowSuffix;
    }
    
    public String getRowSeparator() {
    	return rowSeparator;
    }
    
    public String getColumnSeparator() {
    	return columnSeparator;
    }
    
    /**
     * Get the string that will separate columns. 
     * Generated from the whitespace parameter and the column
     * separating character. Thus, if whitespace is 4, there will be
     * two spaces on either side of the separating character.
     * @return colSepStr
     */
    public String getGeneratedColumnSeparatorString() {
    	return colSepStr;
    }
    
    public NumberFormat getFormat() {
    	return format;
    }
    
    public int getWhitespace() {
    	return whiteSpace;
    }
    
    private static boolean isNumeric(Object o) {
    	return (o instanceof Number);
    }
    
    public void toggleAlignment() {
    	align = align.equals(RIGHT) ? LEFT : RIGHT;
    }
}