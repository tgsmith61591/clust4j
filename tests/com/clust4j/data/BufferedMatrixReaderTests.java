package com.clust4j.data;

import static org.junit.Assert.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.utils.MatUtils;


public class BufferedMatrixReaderTests {
	final static String file = new String("tmpbmrtfile.csv");
	final static Path path = FileSystems.getDefault().getPath(file);
	
	// Mock up a write, then a read...
	@Test
	public void test1() throws FileNotFoundException, IOException, ClassNotFoundException {
		// test simple
		Object[] o = new Object[]{
			"1,2,3,4,5",
			"1,2,3,4,5"
		};
		
		writeCSV(o);
		assertTrue(
			MatUtils.equalsExactly(new double[][]{
				new double[]{1,2,3,4,5},
				new double[]{1,2,3,4,5}
			}, readCSV().getDataRef())
		);
		Files.delete(path);
	}
	
	@Test(expected=FileNotFoundException.class)
	public void test2() throws FileNotFoundException, IOException {
		readCSV().getDataRef();
	}
	
	@Test(expected=FileNotFoundException.class)
	public void test3() throws FileNotFoundException, IOException, ClassNotFoundException {
		SimpleBufferedMatrixReader.readMatrix(new FileInputStream(file));
	}
	
	@Test
	public void test4() throws FileNotFoundException, IOException {
		boolean passes =false;
		
		try {
			// test simple
			Object[] o = new Object[]{
				"1,2,a,4,5",
				"1,2,3,4,5"
			};
			
			writeCSV(o);
			readCSV();
		} catch(NumberFormatException n) {
			passes = true;
		} finally {
			Files.delete(path);
		}
		
		assertTrue(passes);
	}
	
	
	static Array2DRowRealMatrix readCSV() throws FileNotFoundException, IOException {
		return new SimpleBufferedMatrixReader(new File(file), ",").readMatrix();
	}
	
	static void writeCSV(Object[] in) throws IOException {
		final String sep = System.getProperty("line.separator");
		StringBuilder sb = new StringBuilder();
		
		for(int i= 0; i < in.length; i++) {
			sb.append(in[i].toString());
			if(i != in.length - 1)
				sb.append(",");
			sb.append(sep);
		}
		
		String out = sb.toString();
		
		// Actually do the writing...
		BufferedWriter bf = new BufferedWriter(new FileWriter(file));
		try {
			bf.write(out);
		} finally {
			bf.close();
		}
	}
}
