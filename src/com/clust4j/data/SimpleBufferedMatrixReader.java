package com.clust4j.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.ObjectInputStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * A utility class solely for internal use 
 * in creating new datasets.
 * @author Taylor G Smith
 */
class SimpleBufferedMatrixReader {
	final File file;
	final String separator;
	
	public SimpleBufferedMatrixReader(final File file, final String separator) throws FileNotFoundException {
		if(!file.exists())
			throw new FileNotFoundException();
		
		this.file = file;
		this.separator = separator;
	}
	
	static long countLines(final File file) throws IOException {
		long lines = 0;
		
		LineNumberReader lnr = null;
		try {
			lnr = new LineNumberReader(new FileReader(file));
			lnr.skip(Long.MAX_VALUE);
			lines = lnr.getLineNumber();
		} finally {
			try {
				lnr.close();
			} catch(NullPointerException n) {
				// Only happens if lnr wasn't initialized anyways...
			}
		}
		
		return lines;
	}
	
	public static Array2DRowRealMatrix readMatrix(FileInputStream fis) throws IOException, ClassNotFoundException {
		ObjectInputStream in = null;
		Array2DRowRealMatrix res;
		
		try {
			in = new ObjectInputStream(fis);
	        res = (Array2DRowRealMatrix) in.readObject();
		} finally {
			try {
				in.close();
			} catch(NullPointerException n) {
				// only happens if improperly initialized...
			}
				
	        fis.close();
		}
        
        return res;
	}
	
	public Array2DRowRealMatrix readMatrix() throws IOException {
		final long lines = countLines(file);
		if(lines > Integer.MAX_VALUE)
			throw new IllegalArgumentException("matrix is larger than acceptable size!");
		
		final double[][] matrix = new double[(int)lines][];
		BufferedReader br = new BufferedReader(new FileReader(file));
		
		String line;
		String[] row;
		double[] newRow;
		int idx = 0;
		
		try {
			while(null != (line = br.readLine())) {
				row = line.split(separator);
				newRow = new double[row.length];
				
				for(int i = 0; i < newRow.length; i++) {
					try {
						newRow[i] = Double.valueOf(row[i]);
					} catch(NumberFormatException nfe) {
						throw new NumberFormatException("format exception for " 
							+ row[i] + " in row " + (idx + 1));
					}
				}
				
				matrix[idx++] = newRow;
			}
		} finally {
			// always close..
			br.close();
		}
		
		return new Array2DRowRealMatrix(matrix, false);
	}
}
