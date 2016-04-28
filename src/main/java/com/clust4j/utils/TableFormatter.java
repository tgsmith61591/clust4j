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

import static com.clust4j.utils.TableFormatter.ColumnAlignment.LEFT;
import static com.clust4j.utils.TableFormatter.ColumnAlignment.RIGHT;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

import org.apache.commons.math3.util.FastMath;

public class TableFormatter implements java.io.Serializable {
	private static final long serialVersionUID = -4944015740188846236L;

	public static enum ColumnAlignment implements java.io.Serializable {
		LEFT {
			@Override
			String justify(String appender, String f) {
				StringBuilder entry = new StringBuilder();
				entry.append(f);
				entry.append(appender);
				return entry.toString();
			}
		}, 
		
		RIGHT {
			@Override
			String justify(String appender, String f) {
				StringBuilder entry = new StringBuilder();
				entry.append(appender);
				entry.append(f);
				return entry.toString();
			}
		}
		;
		
		abstract String justify(String appender, String f);
	}
	
    /** The default prefix: "". */
	public static final String DEFAULT_PREFIX = "";
    /** The default suffix: "". */
	public static final String DEFAULT_SUFFIX = "";
    /** The default row prefix: "". */
	public static final String DEFAULT_ROW_PREFIX = "";
    /** The default row suffix: "". */
	public static final String DEFAULT_ROW_SUFFIX = "";
    /** The default row separator: "\n". */
	public static final String DEFAULT_ROW_SEPARATOR = System.getProperty("line.separator");
    /** The default column separator: "". */
	public static final String DEFAULT_COLUMN_SEPARATOR = "";
    /** The default locale */
    public static final Locale DEFAULT_LOCALE = Locale.US;
    /** The default number format */
    public static final NumberFormat DEFAULT_NUMBER_FORMAT = NumberFormat.getInstance(DEFAULT_LOCALE); 
    /** The default whitespace between columns */
    public static final int DEFAULT_WHITE_SPACE = 4;
    /** Default column alignment */
    public static final ColumnAlignment DEFAULT_ALIGNMENT = RIGHT;
    static final String NULL_STR = "--";
    static final int MIN_WIDTH = 3;

    
    /*
     * Don't want 100k getters/setters for these...
     */
    /** Prefix. */
    public String prefix;
    /** Suffix. */
    public String suffix;
    /** Row prefix. */
    public String rowPrefix;
    /** Row suffix. */
    public String rowSuffix;
    /** Row separator. */
    public String rowSeparator;
    /** Column separator. */
    public String columnSeparator;
    /** The format used for components. */
    public NumberFormat format;
    
    
    /** The whitespace between cols */
    protected int whiteSpace;
	/** Line sep */
    protected final String lineSep;
	/** Between col string constructed from whitespace and char */
    protected String colSepStr;
	/** Column alignment for printing */
	protected ColumnAlignment align = DEFAULT_ALIGNMENT;
	public boolean leadWithEmpty = true;
	public char tableBreakChar = '.';
	
	
	
    public TableFormatter() {
    	this(DEFAULT_NUMBER_FORMAT);
    }
    
    public TableFormatter(final ColumnAlignment align) {
    	this();
    	this.align = align;
    }
    
    public TableFormatter(final NumberFormat format) {
    	this(DEFAULT_PREFIX, DEFAULT_SUFFIX, DEFAULT_ROW_PREFIX, DEFAULT_ROW_SUFFIX, 
    		DEFAULT_ROW_SEPARATOR, DEFAULT_COLUMN_SEPARATOR, DEFAULT_WHITE_SPACE, format);
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
    	setWhiteSpace(whiteSpace);
    	
    	this.format = format;
    	this.lineSep = System.getProperty("line.separator");
    }
    
    
    public class Table {
    	final private String fmt;
    	/**
    	 * Some tables are very long and need a break in the middle.
    	 * After a format, this will generate the appropriate table break.
    	 */
    	private String tableBreak;
    	
    	
    	Table(ArrayList<Object[]> matrix, int numRows) {
    		this.fmt = fmt(matrix, numRows);
    	}
    	
		private String fmt(ArrayList<Object[]> matrix, int numRows) {
			final int rows = matrix.size();
			if (numRows < 1)
				throw new IllegalArgumentException("numrows must exceed 0");
			else if (numRows > rows)
				numRows = rows;

			StringBuilder output = new StringBuilder();
			output.append(prefix + (leadWithEmpty ? lineSep : ""));

			final Object[][] data = new Object[numRows][];
			for (int i = 0; i < numRows; i++) {
				Object[] matI = matrix.get(i);
				data[i] = matI;
			}

			// Get the max num columns...
			int largestSoFar = Integer.MIN_VALUE;
			for (Object[] oo : data)
				if (oo.length > largestSoFar)
					largestSoFar = oo.length;

			// Assign as max...
			final int cols = largestSoFar;

			/* While finding width, go ahead and format */
			final String[][] formatted = new String[numRows][cols];

			// Need to get the max width for each column
			ArrayList<Integer> idxToWidth = new ArrayList<Integer>(cols);
			for (int col = 0; col < cols; col++) {
				int maxWidth = Integer.MIN_VALUE;
				for (int row = 0; row < numRows; row++) {
					String f;
					int len;

					if (data[row].length <= col) {
						f = ""; // Set to empty, if not exists...
					} else {
						f = formatNumber(data[row][col]);
					}

					len = f.length();
					if (len > maxWidth)
						maxWidth = len;
					formatted[row][col] = f;
				}

				idxToWidth.add(FastMath.max(maxWidth, MIN_WIDTH));
			}

			// Now append plus width, etc.
			for (int row = 0; row < numRows; row++) {

				// Build the break formatter if the first iteration...
				if (0 == row) {
					StringBuilder linebreak = new StringBuilder();
					linebreak.append(rowPrefix);

					for (int col = 0; col < cols; col++) {
						StringBuilder entry = new StringBuilder();

						char[] filler = new char[MIN_WIDTH];
						Arrays.fill(filler, tableBreakChar);
						String f = new String(filler);

						int len = f.length();
						int colMaxLen = idxToWidth.get(col);
						int def = colMaxLen - len;
						String appender = getAppenderOfLen(def);

						entry.append(align.justify(appender, f));
						linebreak.append(entry.toString()
								+ (col == cols - 1 ? rowSuffix + lineSep
										: colSepStr));
					}

					this.tableBreak = linebreak.toString();
				}

				StringBuilder rowBuild = new StringBuilder();
				rowBuild.append(rowPrefix);
				for (int col = 0; col < cols; col++) {
					StringBuilder entry = new StringBuilder();
					String f = formatted[row][col];
					int len = f.length();
					int colMaxLen = idxToWidth.get(col);
					int deficit = colMaxLen - len;
					String appender = getAppenderOfLen(deficit);

					entry.append(align.justify(appender, f));
					rowBuild.append(entry.toString()
							+ (col == cols - 1 ? rowSuffix + lineSep
									: colSepStr));
				}

				output.append(rowBuild);
			}

			output.append(suffix);
			return output.toString();
		}
	    
	    String formatNumber(Object o) {
	    	String f;
	    	if(isNumeric(o)) {
				Double tmpd = ((Number)o).doubleValue();
				f = tmpd.isNaN() ? "NaN" : 
					tmpd.isInfinite() ? (tmpd.equals(Double.NEGATIVE_INFINITY) ? "-Inf" : "Inf") : 
						format.format(tmpd);
			} else {
				f = null == o ? 
					"null" : o.toString();
			}
	    	
	    	return f;
	    }
	    
	    /**
		 * Some tables are very long and need a break in the middle.
		 * After a format, this will generate the appropriate table break.
		 * Otherwise, it will return null.
		 */
	    public String getTableBreak() {
	    	return tableBreak;
	    }
    	
    	@Override
    	public String toString() {
    		return fmt;
    	}
    }
    
    
    public Table format(ArrayList<Object[]> rows) {
    	return format(rows, rows.size());
    }
    
    public Table format(ArrayList<Object[]> matrix, int numRows) {
    	return new Table(matrix, numRows);
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
    
    public int getWhitespace() {
    	return whiteSpace;
    }
    
    private static boolean isNumeric(Object o) {
    	if(null == o)
    		return false;
    	return (o instanceof Number);
    }
    
    public void setWhiteSpace(int n) {
    	this.whiteSpace = n%2==0 ? n : n + 1;
    	String ws = getAppenderOfLen(this.whiteSpace/2);
    	this.colSepStr = ws + this.columnSeparator + ws; //"  |  " or the likes
    }
    
    public void toggleAlignment() {
    	align = align.equals(RIGHT) ? LEFT : RIGHT;
    }
}