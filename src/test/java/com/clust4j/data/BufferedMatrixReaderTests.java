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

import static org.junit.Assert.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.data.BufferedMatrixReader.MatrixReaderSetup;
import com.clust4j.except.MatrixParseException;
import com.clust4j.utils.MatUtils;


public class BufferedMatrixReaderTests {
	final static boolean PARALLEL = GlobalState.ParallelismConf.PARALLELISM_ALLOWED;
	final static String file = new String("tmpbmrtfile.csv");
	final static Path path = FileSystems.getDefault().getPath(file);
	final static byte HIVE = (byte)0x1;
	
	static Object[] fromDoubleArr(double[][] d) {
		final Object[] o = new Object[d.length];
		
		int idx = 0;
		String s;
		for(double[] dub: d) {
			s = Arrays.toString(dub);
			s = s.substring(1, s.length() - 1);
			o[idx++] = s;
		}
		
		return o;
	}
	
	static DataSet readCSV(boolean b) throws FileNotFoundException, IOException {
		return new BufferedMatrixReader(new File(file)).read(b);
	}
	
	static DataSet readCSV() throws FileNotFoundException, IOException {
		return new BufferedMatrixReader(new File(file)).read();
	}
	
	static void writeCSV(Object[] in) throws IOException {
		final String sep = System.getProperty("line.separator");
		StringBuilder sb = new StringBuilder();
		
		for(int i= 0; i < in.length; i++) {
			sb.append(in[i].toString());
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
	
	static void writeCSVHiveSep(double[][] in) throws IOException {
		final String sep = System.getProperty("line.separator");
		StringBuilder sb = new StringBuilder();
		final char hive = (char)HIVE;
		
		for(int i= 0; i < in.length; i++) {
			StringBuilder row = new StringBuilder();
			
			for(int j= 0; j < in[i].length; j++) {
				row.append(in[i][j]);
				if(j != in[i].length - 1)
					row.append(hive);
			}
			
			sb.append(row.toString());
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
	
	@Test
	public void testBuffered1() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{
				"1,2,3,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5},
					new double[]{6,7,8,9,10}
				}, readCSV().getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testBufferedCustomDelim() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{
				"1$2$3$4$5",
				"6$7$8$9$10"
			};
			
			
			writeCSV(o);
			DataSet d = new BufferedMatrixReader(new File(file), (byte)'$').read();
			
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testBufferedSingleQuoted() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{
				"'1','2','3','4','5'",
				"'6','7','8','9','10'"
			};
			
			writeCSV(o);
			DataSet d = new BufferedMatrixReader(new File(file), true).read();
			
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testBufferedDoubleQuoted() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{
				"\"1\",\"2\",\"3\",\"4\",\"5\"",
				"\"6\",\"7\",\"8\",\"9\",\"10\""
			};
			
			writeCSV(o);
			DataSet d = new BufferedMatrixReader(new File(file), false).read();
			
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testBufferedDoubleQuotedUnmatched() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{
				"\"1\",\"2\",\"3\",\"4\",\"5\"",
				"\"6\",\"7\",\"8\",\"9\",10"
			};
			
			writeCSV(o);
			DataSet d = new BufferedMatrixReader(new File(file), false).read();
			
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testTrailingComma1() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,3,4,5,",
				"6,7,8,9,10,"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5,Double.NaN},
					new double[]{6,7,8,9,10,Double.NaN}
				}, d.getDataRef().getDataRef())
			);
			
			assertTrue(MatUtils.containsNaN(d.getDataRef().getDataRef()));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testInf1() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,inf,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,Double.POSITIVE_INFINITY,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testInf2() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,infinity,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,Double.POSITIVE_INFINITY,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testNegInf1() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,-inf,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,Double.NEGATIVE_INFINITY,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testNegInf2() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,-infinity,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,Double.NEGATIVE_INFINITY,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testNaNEmbedded1() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,nan,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(MatUtils.containsNaN(d.getDataRef().getDataRef()));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testNaNEmbedded2() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2, ,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(MatUtils.containsNaN(d.getDataRef().getDataRef()));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testNaNEmbedded3() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(MatUtils.containsNaN(d.getDataRef().getDataRef()));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testNaNEmbedded4() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,na,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(MatUtils.containsNaN(d.getDataRef().getDataRef()));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testNaNEmbedded5() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,?,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			DataSet d= readCSV();
			assertTrue(MatUtils.containsNaN(d.getDataRef().getDataRef()));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	/*
	 * Test that merely the presence of a ? is not enough to trigger NaN
	 */
	@Test(expected=MatrixParseException.class)
	public void testEmbeddedNonNaN() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,a?,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testDifferentDelim() throws IOException {
		try {
			Object[] o = new Object[]{
				"1|2|3|4|5",
				"6|7|8|9|10"
			};
			
			writeCSV(o);
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5},
					new double[]{6,7,8,9,10}
				}, readCSV().getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testWithHeader() throws IOException {
		try {
			Object[] o = new Object[]{
				"a|b|c|d|e",
				"1|2|3|4|5",
				"6|7|8|9|10"
			};
			
			writeCSV(o);
			DataSet d = readCSV();
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5},
					new double[]{6,7,8,9,10}
				}, d.getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testLarge() throws IOException {
		try {
			double[][] d = MatUtils.randomGaussian(5000, 25);
			
			Object[] o = fromDoubleArr(d);
			writeCSV(o);
			assertTrue(
				MatUtils.equalsExactly(d, 
					readCSV().getDataRef().getDataRef())
			);
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testParseException1() throws IOException {
		/*
		 * This one will fail in the setup
		 */
		try {
			Object[] o = new Object[]{
				"1|2|a|4|5",
				"6|7|8|9|10"
			};
			
			writeCSV(o);
			readCSV();
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testParseException2() throws IOException {
		/*
		 * This one will fail on read()
		 */
		try {
			Object[] o = new Object[]{
				"1|2|3|4|5",
				"6|7|8|9|10",
				"1|2|3|4|5",
				"6|7|8|9|10",
				"1|2|3|4|5",
				"6|7|8|9|10",
				"1|2|3|4|5",
				"6|7|a|9|10",
			};
			
			writeCSV(o);
			readCSV();
			
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testSingleRow() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{ "1,2,3,4,5" };
			
			writeCSV(o);
			BufferedMatrixReader bmr = new BufferedMatrixReader(new File(file));
			DataSet d = bmr.read();
			assertTrue(bmr.setup.header_offset == 0);
			assertTrue(bmr.setup.headers == null);
			
			assertTrue(
				MatUtils.equalsExactly(new double[][]{
					new double[]{1,2,3,4,5}
				}, d.getDataRef().getDataRef())
			);
			
			assertTrue(bmr.hasWarnings());
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testSingleRowNumericFailure() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{ "1,2,a,4,5" };
			
			writeCSV(o);
			readCSV();
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testSingleRowHeaderFailure() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{ "a,b,c,d,e" };
			
			writeCSV(o);
			readCSV();
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testSingleRowNoSep() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{ "abcde" };
			
			writeCSV(o);
			readCSV();
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testJagged() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{
					"1,2,3,4,5",
					"6,7,8",
					"1,2,3,4,5"
				};
			
			writeCSV(o);
			readCSV();
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testPrettyPrinterInError() throws IOException {
		// test simple
		try {
			Object[] o = new Object[]{
				"1,2,3,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z"
			};
			
			writeCSV(o);
			readCSV();
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testParallel1() throws IOException {
		try {
			double[][] g = MatUtils.randomGaussian(50, 5);
			Object[] o = fromDoubleArr(g);
			
			writeCSV(o);
			DataSet d = readCSV(PARALLEL);
			
			assertTrue(MatUtils.equalsExactly(g, d.getDataRef().getDataRef()));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testParallelBig() throws IOException {
		double[][] g = MatUtils.randomGaussian(500, 150); // make smaller for travis CI
		Object[] o = fromDoubleArr(g);
		writeCSV(o);
		
		try {
			for(boolean parallel: new boolean[]{false, PARALLEL}) {
					System.out.println((parallel?"Parallel":"Serial")+" parsing task");
					DataSet d = readCSV(parallel);
					
					assertTrue(MatUtils.equalsExactly(g, d.getDataRef().getDataRef()));
					System.out.println();
			}

		} finally {
			Files.delete(path);
		}
		
	}
	
	@Test(expected=MatrixParseException.class)
	public void testParallelBigNFE1() throws IOException {
		double[][] g = MatUtils.randomGaussian(500, 150); // make smaller for travis CI
		Object[] o = fromDoubleArr(g);
		o[15] = new Object[]{"asdf"};
		writeCSV(o);
		
		try {
			readCSV(PARALLEL);
			System.out.println();

		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testParallelBigDME1() throws IOException {
		double[][] g = MatUtils.randomGaussian(500, 150); // make smaller for travis CI
		Object[] o = fromDoubleArr(g);
		o[15] = new Integer(1);
		writeCSV(o);
		
		try {
			readCSV(PARALLEL);
			System.out.println();

		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testLeadingAndTrailingCommentsAndEmptyLines() throws IOException {
		double[][] g = MatUtils.randomGaussian(100, 150);
		
		double[][] gp = new double[g.length - 3][g[0].length];
		for(int i = 1, idx = 0; i < g.length - 2; i++, idx++)
			gp[idx] = g[i];
		
		
		Object[] o = fromDoubleArr(g);
		o[0] = "# leading comment";
		o[g.length - 2] = "";
		o[g.length - 1] = "# trailing comment";
		writeCSV(o);
		
		try {
			double[][] gpp = readCSV(PARALLEL).getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(gp, gpp));
			System.out.println();

		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testDifferentConstructors() throws IOException {
		double[][] g = MatUtils.randomGaussian(10, 10);
		Object[] o = fromDoubleArr(g);
		writeCSV(o);
		
		try {
			File f = new File(file);
			
			double[][] d = new BufferedMatrixReader(f, true).read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(f, false).read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(f, (byte)',').read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(f, true, (byte)',').read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(f, false, (byte)',').read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			/*
			 * Now let's read all the bytes from the file first...
			 */
			byte[] bits = BufferedMatrixReader.fileToBytes(f);
			
			d = new BufferedMatrixReader(bits).read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(bits, true).read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(bits, false).read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(bits, true, (byte)',').read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(bits, false, (byte)',').read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			d = new BufferedMatrixReader(bits, (byte)',').read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testHiveSep() throws IOException {
		double[][] g = MatUtils.randomGaussian(10, 10);
		writeCSVHiveSep(g);
		
		try {
			// Test auto-detect
			double[][] d = new BufferedMatrixReader(new File(file)).read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
			
			// test provide
			d = new BufferedMatrixReader(new File(file), (byte)0x1).read().getDataRef().getDataRef();
			assertTrue(MatUtils.equalsExactly(d, g));
			System.out.println();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testEmptySet() throws IOException {
		writeCSV(new Object[]{});
		
		try {
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testOddSeparator() throws IOException {
		writeCSV(new Object[]{
			"1,001 5,002 12",	// first and second records need to have differing num ','
			"3,253 2,162 9,102",// second and third records need to have differing num ','
			"6,019 194 9,274",	// third and first records need to have same num ','
		});
		
		try {
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testNoGoodSep() throws IOException {
		writeCSV(new Object[]{
			"123 123 123",
			"123,123,123"
		});
		
		try {
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void testMaxLen() throws IOException {
		/*
		try {
			double[][] d = new double[com.clust4j.GlobalState.MAX_ARRAY_SIZE][];
			for(int i = 0; i < d.length; i++)
				d[i] = new double[]{0,1};
			
			Object[] o = fromDoubleArr(d);
			d = null; // force GC
			
			
			writeCSV(o);
			try {
				BufferedMatrixReader bmr = new BufferedMatrixReader(new File(file));
				bmr.read(true);
				assertTrue(bmr.hasWarnings());
			} finally {
				Files.delete(path);
			}
		} catch(OutOfMemoryError o) {
			o.printStackTrace();
		}
		*/
	}
	
	@Test
	public void testSetupCopy() throws IOException {
		try {
			Object[] o = new Object[]{
				"1,2,3,4,5",
				"6,7,8,9,10"
			};
			
			writeCSV(o);
			final byte[] original = BufferedMatrixReader.fileToBytes(new File(file));
			final MatrixReaderSetup mrs = new MatrixReaderSetup(original);
			
			final MatrixReaderSetup copy= mrs.copy();
			original[0] = HIVE;
			assertFalse(copy.stream[0] == original[0]); // assert not the same reference
			
			mrs.headers = new String[]{"asdf","asdf","asdf","asdf","asdf"};
			assertNull(copy.headers); // assert not the same reference
			
			mrs.data = null;
			assertNotNull(copy.data); // assert not same ref
			
			writeCSV(o);
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testEmpty() throws IOException {
		try {
			Object[] o = new Object[]{
				""
			};
			
			writeCSV(o);
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testOneRowNoConventionalSep() throws IOException {
		try {
			Object[] o = new Object[]{
				"1*1*1"
			};
			
			writeCSV(o);
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testOrphanedHeader() throws IOException {
		try {
			Object[] o = new Object[]{
				"column_a column_b column_c"
			};
			
			writeCSV(o);
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testInconsistentSeparator() throws IOException {
		try {
			Object[] o = new Object[]{
				"1 1 1",
				"2,2,2",
				"3;3;3"
			};
			
			writeCSV(o);
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test(expected=MatrixParseException.class)
	public void testLurkingAlpha() throws IOException {
		try {
			Object[] o = new Object[]{
				"1 1 1",
				"2 a 2",
				"3 3 3"
			};
			
			writeCSV(o);
			readCSV();
		} finally {
			Files.delete(path);
		}
	}
	
	@Test
	public void coverageLove() throws IOException {
		try {
			Object[] o = new Object[]{
				"1 1 1",
				"2 2 2",
				"3 3 3"
			};
			
			writeCSV(o);
			byte[] bits = BufferedMatrixReader.fileToBytes(new File(file));
			
			MatrixReaderSetup mrs = new MatrixReaderSetup(bits);
			mrs.debug("coverage love!");
			mrs.trace("coverage love!");
			
			BufferedMatrixReader bmr = new BufferedMatrixReader(mrs);
			bmr.warn("coverage warn!");
			bmr.trace("coverage love!");
			bmr.debug("coverage love!");
			
			try {
				bmr.error(new RuntimeException("coverage error!"));
			} catch(RuntimeException r) {
				// expected this.
			}
			
		} finally {
			Files.delete(path);
		}
	}
}
