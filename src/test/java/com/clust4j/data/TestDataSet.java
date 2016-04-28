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

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.AffinityPropagation;
import com.clust4j.algo.AffinityPropagationParameters;
import com.clust4j.algo.DBSCAN;
import com.clust4j.algo.DBSCANParameters;
import com.clust4j.algo.HDBSCAN;
import com.clust4j.algo.HDBSCANParameters;
import com.clust4j.algo.HierarchicalAgglomerative;
import com.clust4j.algo.HierarchicalAgglomerativeParameters;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.KMedoidsParameters;
import com.clust4j.algo.MeanShift;
import com.clust4j.algo.MeanShiftParameters;
import com.clust4j.algo.UnsupervisedClassifier;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.VecUtils;

public class TestDataSet 
		implements java.io.Serializable {
	/** This needs to be serializable for the anonymous class test */
	private static final long serialVersionUID = 4724815970248166429L;
	
	
	private final static DecimalFormat df = new DecimalFormat("##.##");
	private final static DataSet IRIS = ExampleDataSets.loadIris();
	private final static DataSet WINE = ExampleDataSets.loadWine();
	private final static DataSet BC = ExampleDataSets.loadBreastCancer();

	@Test(expected=IllegalStateException.class)
	public void testIris() {
		final int len = IRIS.getDataRef().getRowDimension();
		DataSet shuffled = IRIS.shuffle();
		assertTrue(shuffled.getDataRef().getRowDimension() == len);
		
		// Test that no reference carried over...
		shuffled.getHeaderRef()[0] = "TESTING!";
		assertTrue( !IRIS.getHeaderRef()[0].equals(shuffled.getHeaderRef()[0]) );
		
		shuffled.setColumn("TESTING!", 
			VecUtils.rep(Double.POSITIVE_INFINITY, shuffled.numRows()));
		assertTrue(VecUtils.unique(shuffled.getColumn("TESTING!")).size() == 1);
		
		// Test piecewise col drops
		shuffled.dropCol("TESTING!");
		assertTrue(shuffled.numCols() == 3);
		
		shuffled.dropCol("Sepal Width");
		assertTrue(shuffled.numCols() == 2);
		
		shuffled.dropCol("Petal Length");
		assertTrue(shuffled.numCols() == 1);
		
		// Prepare for the throw...
		shuffled.dropCol("Petal Width"); // BOOM!
	}
	
	
	
	private static String formatPct(double num) {
		return df.format(num * 100) + "%";
	}
	
	private static void stdout(UnsupervisedClassifier model, boolean b, int[] labels) {
		String nm = ((AbstractClusterer)model).getName();
		System.out.println(nm+" (scale = "+b+"):  " + formatPct(model.indexAffinityScore(labels)) );
	}
	
	private static void stdout(UnsupervisedClassifier model, boolean b) {
		String nm = ((AbstractClusterer)model).getName();
		System.out.println(nm+" (scale = "+b+"):  " + model.silhouetteScore() );
	}

	private void testAlgos(boolean shuffle, DataSet ds, int k) {
		System.out.println(" ========== " + "Testing with shuffle" + 
				(shuffle ? " enabled" : " disabled") + 
				" ========== ");
		
		DataSet shuffled = shuffle ? ds.shuffle() : ds;
		int[] labels = shuffled.getLabels();
		
		final Array2DRowRealMatrix data = shuffled.getData();
		StandardScaler scaler = new StandardScaler().fit(data);
		RealMatrix X = scaler.transform(data);
		//final int[] actual = shuffled.getLabels();
		
		final boolean verbose = false;
		final boolean[] scale = new boolean[]{false, true};
		for(boolean b: scale) {
			
			// Go down the line alpha...
			AffinityPropagation ap = new AffinityPropagationParameters()
					.setVerbose(verbose)
					.fitNewModel(b ? X : data);
			stdout(ap, b);
			
			
			DBSCAN db = new DBSCANParameters()
					.setVerbose(verbose)
					.fitNewModel(b ? X : data);
			stdout(db, b);
			
			HDBSCAN hdb = new HDBSCANParameters()
					.setVerbose(verbose)
					.fitNewModel(b ? X : data);
			stdout(hdb, b);
			
			HierarchicalAgglomerative ha = new HierarchicalAgglomerativeParameters()
					.setVerbose(verbose)
					.fitNewModel(b ? X : data);
			stdout(ha, b);
			
			
			KMeans kmn = new KMeansParameters(k)
					.setVerbose(verbose)
					.fitNewModel(b ? X : data);
			stdout(kmn, b, labels);
			
			KMedoids kmd = new KMedoidsParameters(k)
					.setVerbose(verbose)
					.fitNewModel(b ? X : data);
			stdout(kmd, b, labels);
			
			MeanShift ms = new MeanShiftParameters()
					.setVerbose(verbose)
					.fitNewModel(b ? X : data);
			stdout(ms, b);
			
			
			System.out.println();
		}
		
		System.out.println();
	}
	
	@Test
	public void testDifferentAlgorithmWithShuffleIRIS() {
		testAlgos(true, IRIS, 3);
	}
	
	@Test
	public void testDifferentAlgorithmNoShuffleIRIS() {
		testAlgos(false, IRIS, 3);
	}
	
	@Test
	public void testDifferentAlgorithmWithShuffleWINE() {
		testAlgos(true, WINE, 3);
	}
	
	@Test
	public void testDifferentAlgorithmNoShuffleWINE() {
		testAlgos(false, WINE, 3);
	}
	
	@Test
	public void testDifferentAlgorithmWithShuffleBC() {
		testAlgos(true, BC, 2);
	}
	
	@Test
	public void testDifferentAlgorithmNoShuffleBC() {
		testAlgos(false, BC, 2);
	}
	
	@Test
	public void testCopyIRIS() {
		DataSet iris = IRIS;
		DataSet shuffle = iris.copy();
		assertTrue(iris.equals(shuffle));
		assertFalse(iris == shuffle);
		
		shuffle = shuffle.shuffle();
		assertFalse(shuffle.equals(iris));
		
		assertFalse(shuffle.getDataRef().equals(iris.getDataRef()));
		assertFalse(shuffle.getHeaderRef().equals(iris.getHeaderRef()));
		assertFalse(shuffle.getLabelRef().equals(iris.getLabelRef()));
	}
	
	@Test
	public void testCopyWINE() {
		DataSet data = WINE;
		DataSet shuffle = data.copy();
		assertTrue(data.equals(shuffle));
		assertFalse(data == shuffle);
		
		shuffle = shuffle.shuffle();
		assertFalse(shuffle.equals(data));
		
		assertFalse(shuffle.getDataRef().equals(data.getDataRef()));
		assertFalse(shuffle.getHeaderRef().equals(data.getHeaderRef()));
		assertFalse(shuffle.getLabelRef().equals(data.getLabelRef()));
	}
	
	@Test
	public void testDataSetColAddsRemoves() {
		DataSet iris = IRIS;
		DataSet shuffle = iris.copy();
		
		final int m = shuffle.getDataRef().getRowDimension();
		final int n = shuffle.getDataRef().getColumnDimension();
		
		double[] newCol = VecUtils.randomGaussian(m);
		shuffle.addColumn(newCol);
		assertTrue(shuffle.getHeaderRef()[n].equals("V" + n));
		
		newCol = VecUtils.randomGaussian(m);
		shuffle.addColumn("NewCol", newCol);
		assertTrue(shuffle.getHeaderRef()[n + 1].equals("NewCol"));
		
		double[][] newCols = MatUtils.randomGaussian(m, 3);
		shuffle.addColumns(newCols);
		assertTrue(shuffle.getDataRef().getColumnDimension() == 9);
		assertTrue(shuffle.getHeaderRef()[8].equals("V8"));
		
		shuffle.dropCol("V4");
		
		//Create a new duplicate named col, capture first val, drop that col
		//(it will drop the first one named that, so the already existing col, 
		//not the new one) and assert the new one remains
		newCol = VecUtils.randomGaussian(m);
		double val = newCol[0];
		shuffle.addColumn("V8", newCol);
		shuffle.dropCol("V8");
		assertTrue(shuffle.getColumn("V8")[0] == val);
		
		// Explicitly test the setCol method
		newCol = VecUtils.randomGaussian(m);
		val = newCol[0];
		shuffle.setColumn("V8", newCol);
		assertTrue(shuffle.getColumn("V8")[0] == val);
		
		shuffle.sortAscInPlace("Sepal Length");
		assertTrue(shuffle.getColumn("Sepal Length")[0] == 4.3); // min val

		shuffle.head();
		
		val = shuffle.getColumn("Sepal Length")[shuffle.numRows()-1];
		shuffle.sortDescInPlace("Sepal Length");
		assertTrue(shuffle.getColumn("Sepal Length")[0] == val); // max val
		
		shuffle.head();
	}
	
	@Test
	public void testIrisAccurate() {
		/*
		 * Since so many tests rely on iris and
		 * the comparison to sklearn ops, we need
		 * to ensure iris is, in fact, equal to sklearn's
		 */
		double[][] sklearn = new double[][]{
			new double[]{ 5.1,  3.5,  1.4,  0.2},
			new double[]{ 4.9,  3. ,  1.4,  0.2},
			new double[]{ 4.7,  3.2,  1.3,  0.2},
			new double[]{ 4.6,  3.1,  1.5,  0.2},
			new double[]{ 5. ,  3.6,  1.4,  0.2},
			new double[]{ 5.4,  3.9,  1.7,  0.4},
			new double[]{ 4.6,  3.4,  1.4,  0.3},
			new double[]{ 5. ,  3.4,  1.5,  0.2},
			new double[]{ 4.4,  2.9,  1.4,  0.2},
			new double[]{ 4.9,  3.1,  1.5,  0.1},
			new double[]{ 5.4,  3.7,  1.5,  0.2},
			new double[]{ 4.8,  3.4,  1.6,  0.2},
			new double[]{ 4.8,  3. ,  1.4,  0.1},
			new double[]{ 4.3,  3. ,  1.1,  0.1},
			new double[]{ 5.8,  4. ,  1.2,  0.2},
			new double[]{ 5.7,  4.4,  1.5,  0.4},
			new double[]{ 5.4,  3.9,  1.3,  0.4},
			new double[]{ 5.1,  3.5,  1.4,  0.3},
			new double[]{ 5.7,  3.8,  1.7,  0.3},
			new double[]{ 5.1,  3.8,  1.5,  0.3},
			new double[]{ 5.4,  3.4,  1.7,  0.2},
			new double[]{ 5.1,  3.7,  1.5,  0.4},
			new double[]{ 4.6,  3.6,  1. ,  0.2},
			new double[]{ 5.1,  3.3,  1.7,  0.5},
			new double[]{ 4.8,  3.4,  1.9,  0.2},
			new double[]{ 5. ,  3. ,  1.6,  0.2},
			new double[]{ 5. ,  3.4,  1.6,  0.4},
			new double[]{ 5.2,  3.5,  1.5,  0.2},
			new double[]{ 5.2,  3.4,  1.4,  0.2},
			new double[]{ 4.7,  3.2,  1.6,  0.2},
			new double[]{ 4.8,  3.1,  1.6,  0.2},
			new double[]{ 5.4,  3.4,  1.5,  0.4},
			new double[]{ 5.2,  4.1,  1.5,  0.1},
			new double[]{ 5.5,  4.2,  1.4,  0.2},
			new double[]{ 4.9,  3.1,  1.5,  0.1},
			new double[]{ 5. ,  3.2,  1.2,  0.2},
			new double[]{ 5.5,  3.5,  1.3,  0.2},
			new double[]{ 4.9,  3.1,  1.5,  0.1},
			new double[]{ 4.4,  3. ,  1.3,  0.2},
			new double[]{ 5.1,  3.4,  1.5,  0.2},
			new double[]{ 5. ,  3.5,  1.3,  0.3},
			new double[]{ 4.5,  2.3,  1.3,  0.3},
			new double[]{ 4.4,  3.2,  1.3,  0.2},
			new double[]{ 5. ,  3.5,  1.6,  0.6},
			new double[]{ 5.1,  3.8,  1.9,  0.4},
			new double[]{ 4.8,  3. ,  1.4,  0.3},
			new double[]{ 5.1,  3.8,  1.6,  0.2},
			new double[]{ 4.6,  3.2,  1.4,  0.2},
			new double[]{ 5.3,  3.7,  1.5,  0.2},
			new double[]{ 5. ,  3.3,  1.4,  0.2},
			new double[]{ 7. ,  3.2,  4.7,  1.4},
			new double[]{ 6.4,  3.2,  4.5,  1.5},
			new double[]{ 6.9,  3.1,  4.9,  1.5},
			new double[]{ 5.5,  2.3,  4. ,  1.3},
			new double[]{ 6.5,  2.8,  4.6,  1.5},
			new double[]{ 5.7,  2.8,  4.5,  1.3},
			new double[]{ 6.3,  3.3,  4.7,  1.6},
			new double[]{ 4.9,  2.4,  3.3,  1. },
			new double[]{ 6.6,  2.9,  4.6,  1.3},
			new double[]{ 5.2,  2.7,  3.9,  1.4},
			new double[]{ 5. ,  2. ,  3.5,  1. },
			new double[]{ 5.9,  3. ,  4.2,  1.5},
			new double[]{ 6. ,  2.2,  4. ,  1. },
			new double[]{ 6.1,  2.9,  4.7,  1.4},
			new double[]{ 5.6,  2.9,  3.6,  1.3},
			new double[]{ 6.7,  3.1,  4.4,  1.4},
			new double[]{ 5.6,  3. ,  4.5,  1.5},
			new double[]{ 5.8,  2.7,  4.1,  1. },
			new double[]{ 6.2,  2.2,  4.5,  1.5},
			new double[]{ 5.6,  2.5,  3.9,  1.1},
			new double[]{ 5.9,  3.2,  4.8,  1.8},
			new double[]{ 6.1,  2.8,  4. ,  1.3},
			new double[]{ 6.3,  2.5,  4.9,  1.5},
			new double[]{ 6.1,  2.8,  4.7,  1.2},
			new double[]{ 6.4,  2.9,  4.3,  1.3},
			new double[]{ 6.6,  3. ,  4.4,  1.4},
			new double[]{ 6.8,  2.8,  4.8,  1.4},
			new double[]{ 6.7,  3. ,  5. ,  1.7},
			new double[]{ 6. ,  2.9,  4.5,  1.5},
			new double[]{ 5.7,  2.6,  3.5,  1. },
			new double[]{ 5.5,  2.4,  3.8,  1.1},
			new double[]{ 5.5,  2.4,  3.7,  1. },
			new double[]{ 5.8,  2.7,  3.9,  1.2},
			new double[]{ 6. ,  2.7,  5.1,  1.6},
			new double[]{ 5.4,  3. ,  4.5,  1.5},
			new double[]{ 6. ,  3.4,  4.5,  1.6},
			new double[]{ 6.7,  3.1,  4.7,  1.5},
			new double[]{ 6.3,  2.3,  4.4,  1.3},
			new double[]{ 5.6,  3. ,  4.1,  1.3},
			new double[]{ 5.5,  2.5,  4. ,  1.3},
			new double[]{ 5.5,  2.6,  4.4,  1.2},
			new double[]{ 6.1,  3. ,  4.6,  1.4},
			new double[]{ 5.8,  2.6,  4. ,  1.2},
			new double[]{ 5. ,  2.3,  3.3,  1. },
			new double[]{ 5.6,  2.7,  4.2,  1.3},
			new double[]{ 5.7,  3. ,  4.2,  1.2},
			new double[]{ 5.7,  2.9,  4.2,  1.3},
			new double[]{ 6.2,  2.9,  4.3,  1.3},
			new double[]{ 5.1,  2.5,  3. ,  1.1},
			new double[]{ 5.7,  2.8,  4.1,  1.3},
			new double[]{ 6.3,  3.3,  6. ,  2.5},
			new double[]{ 5.8,  2.7,  5.1,  1.9},
			new double[]{ 7.1,  3. ,  5.9,  2.1},
			new double[]{ 6.3,  2.9,  5.6,  1.8},
			new double[]{ 6.5,  3. ,  5.8,  2.2},
			new double[]{ 7.6,  3. ,  6.6,  2.1},
			new double[]{ 4.9,  2.5,  4.5,  1.7},
			new double[]{ 7.3,  2.9,  6.3,  1.8},
			new double[]{ 6.7,  2.5,  5.8,  1.8},
			new double[]{ 7.2,  3.6,  6.1,  2.5},
			new double[]{ 6.5,  3.2,  5.1,  2. },
			new double[]{ 6.4,  2.7,  5.3,  1.9},
			new double[]{ 6.8,  3. ,  5.5,  2.1},
			new double[]{ 5.7,  2.5,  5. ,  2. },
			new double[]{ 5.8,  2.8,  5.1,  2.4},
			new double[]{ 6.4,  3.2,  5.3,  2.3},
			new double[]{ 6.5,  3. ,  5.5,  1.8},
			new double[]{ 7.7,  3.8,  6.7,  2.2},
			new double[]{ 7.7,  2.6,  6.9,  2.3},
			new double[]{ 6. ,  2.2,  5. ,  1.5},
			new double[]{ 6.9,  3.2,  5.7,  2.3},
			new double[]{ 5.6,  2.8,  4.9,  2. },
			new double[]{ 7.7,  2.8,  6.7,  2. },
			new double[]{ 6.3,  2.7,  4.9,  1.8},
			new double[]{ 6.7,  3.3,  5.7,  2.1},
			new double[]{ 7.2,  3.2,  6. ,  1.8},
			new double[]{ 6.2,  2.8,  4.8,  1.8},
			new double[]{ 6.1,  3. ,  4.9,  1.8},
			new double[]{ 6.4,  2.8,  5.6,  2.1},
			new double[]{ 7.2,  3. ,  5.8,  1.6},
			new double[]{ 7.4,  2.8,  6.1,  1.9},
			new double[]{ 7.9,  3.8,  6.4,  2. },
			new double[]{ 6.4,  2.8,  5.6,  2.2},
			new double[]{ 6.3,  2.8,  5.1,  1.5},
			new double[]{ 6.1,  2.6,  5.6,  1.4},
			new double[]{ 7.7,  3. ,  6.1,  2.3},
			new double[]{ 6.3,  3.4,  5.6,  2.4},
			new double[]{ 6.4,  3.1,  5.5,  1.8},
			new double[]{ 6. ,  3. ,  4.8,  1.8},
			new double[]{ 6.9,  3.1,  5.4,  2.1},
			new double[]{ 6.7,  3.1,  5.6,  2.4},
			new double[]{ 6.9,  3.1,  5.1,  2.3},
			new double[]{ 5.8,  2.7,  5.1,  1.9},
			new double[]{ 6.8,  3.2,  5.9,  2.3},
			new double[]{ 6.7,  3.3,  5.7,  2.5},
			new double[]{ 6.7,  3. ,  5.2,  2.3},
			new double[]{ 6.3,  2.5,  5. ,  1.9},
			new double[]{ 6.5,  3. ,  5.2,  2. },
			new double[]{ 6.2,  3.4,  5.4,  2.3},
			new double[]{ 5.9,  3. ,  5.1,  1.8}
		};
		
		double[][] iris = IRIS.getData().getDataRef();
		assertTrue(MatUtils.equalsExactly(sklearn, iris));
	}
	
	@Test
	public void testConst1() {
		double[][] d = IRIS.getData().getDataRef();
		int[] labs= IRIS.getLabels();
		assertTrue(new DataSet(d, labs).numCols() == 4);
	}
	
	@Test
	public void testConst2() {
		Array2DRowRealMatrix d = IRIS.getData();
		int[] labs= IRIS.getLabels();
		assertTrue(new DataSet(d, labs).numCols() == 4);
	}
	
	@Test
	public void testConst3() {
		Array2DRowRealMatrix d = IRIS.getData();
		int[] labs= IRIS.getLabels();
		assertTrue(new DataSet(d, labs, TestSuite.formatter).numCols() == 4);
	}
	
	@Test
	public void testConst4() {
		Array2DRowRealMatrix d = IRIS.getData();
		int[] labs= IRIS.getLabels();
		DataSet dat = new DataSet(d, labs, new String[]{"a","b","c","d"});
		assertTrue(dat.numCols() == 4);
		assertTrue(VecUtils.equalsExactly(dat.getColumn("a"), d.getColumn(0)));
		assertTrue(VecUtils.equalsExactly(dat.getColumn(0), d.getColumn(0)));
	}
	
	@Test
	public void testConst5() {
		double[][] d = IRIS.getData().getDataRef();
		int[] labs= IRIS.getLabels();
		DataSet dat = new DataSet(d, labs, new String[]{"a","b","c","d"});
		assertTrue(dat.numCols() == 4);
	}
	
	@Test
	public void testConst6() {
		Array2DRowRealMatrix d = IRIS.getData();
		int[] labs= IRIS.getLabels();
		DataSet dat = new DataSet(d, labs, new String[]{"a","b","c","d"}, new MatrixFormatter());
		assertTrue(dat.numCols() == 4);
		assertTrue(VecUtils.equalsExactly(dat.getColumn("a"), d.getColumn(0)));
		assertTrue(VecUtils.equalsExactly(dat.getColumn(0), d.getColumn(0)));
	}
	
	@Test
	public void testConst7() {
		double[][] d = IRIS.getData().getDataRef();
		int[] labs= IRIS.getLabels();
		DataSet dat = new DataSet(d, labs, new String[]{"a","b","c","d"}, new MatrixFormatter());
		assertTrue(dat.numCols() == 4);
	}
	
	@Test
	public void testNullLabels() {
		// make sure we allow this...
		int[] labels = null;
		new DataSet(MatUtils.randomGaussian(5, 2), labels);
	}
	
	@Test
	public void testNullString() {
		// make sure we allow this...
		new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2,3,4,5}, null, null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testNullData() {
		double[][] d = null;
		new DataSet(d, new int[]{1,2,3,4,5});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testLabDim() {
		new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testHeaderDim() {
		new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2,3,4,5}, new String[]{"s"});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testColAddDim() {
		DataSet d = new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2,3,4,5});
		d.addColumn(new double[]{1});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testColsAddDim() {
		DataSet d = new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2,3,4,5});
		double[][] dub = new double[][]{
			new double[]{1,2},
			new double[]{1,2}
		};
		
		d.addColumns(dub);
	}
	
	@Test
	public void testColsAdd() {
		DataSet d = new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2,3,4,5});
		double[][] dub = MatUtils.randomGaussian(5, 2);
		d.addColumns(null, dub);
		assertTrue(d.numCols() == 4);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testBadDrop() {
		DataSet d = new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2,3,4,5});
		d.dropCol(-1);
	}
	
	@Test(expected=IllegalStateException.class)
	public void testBadDrop2() {
		DataSet d = new DataSet(MatUtils.randomGaussian(5, 2), new int[]{1,2,3,4,5});
		while(d.numCols() > 1)
			d.dropCol(0);
		d.dropCol(0);
	}
	
	@Test
	public void testCopyEquals() {
		DataSet iris = IRIS.copy();
		assertTrue(iris.equals(iris));
		
		DataSet irisb= IRIS.copy();
		assertTrue(iris.equals(irisb));
		
		// test not mutable
		irisb.dropCol(0);
		assertFalse(iris.equals(irisb));
		assertFalse(iris.equals(new Object()));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMissingHeader() {
		IRIS.getColumn("missing column");
	}
	
	@Test
	public void testHash() {
		IRIS.hashCode(); // no test... just ensure not null
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testSetColOOB() {
		IRIS.setColumn(9, new double[]{});
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testSetRowOOB() {
		IRIS.setRow(-1, new double[]{});
	}
	
	@Test
	public void testSetRowValid() {
		IRIS.copy().setRow(0, new double[]{1,2,3,4});
	}
	
	@Test
	public void testSort() {
		DataSet i = IRIS.copy();
		i.sortAscInPlace(0);
		assertTrue(i.getColumn(0)[0] <= i.getColumn(0)[1]);
	}
	
	@Test
	public void testSortDesc() {
		DataSet i = IRIS.copy();
		i.sortDescInPlace(0);
		assertTrue(i.getColumn(0)[0] >= i.getColumn(0)[1]);
		
		// stdout now
		i.stdOut();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testSortEx() {
		DataSet i = IRIS.copy();
		i.sortAscInPlace(-1);
	}

	@Test(expected=IllegalArgumentException.class)
	public void testSortDescEx() {
		DataSet i = IRIS.copy();
		i.sortDescInPlace(6);
	}
	
	@Test
	public void testSetLabel1() {
		DataSet i = IRIS.copy();
		i.setLabels(null);
		assertNull(i.getLabels());
	}
	
	@Test
	public void testSetLabel2() {
		DataSet i = IRIS.copy();
		i.setLabels(new int[150]);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testSetLabel3() {
		DataSet i = IRIS.copy();
		i.setLabels(new int[15]);
	}
	
	@Test
	public void ensureShuffleImmutability() {
		DataSet d = IRIS.copy();
		int[] label_ref = d.getLabelRef();
		double[][] mat_ref = d.getDataRef().getDataRef();
		
		DataSet p = d.shuffle();
		p.getLabelRef()[0] = 1_000_000;
		assertFalse(p.getLabelRef()[0] == label_ref[0]);
		
		p.getDataRef().getDataRef()[0][0] = 1_000_000;
		assertFalse(mat_ref[0][0] == p.getDataRef().getDataRef()[0][0]);
	}
	
	@Test
	public void testWrite1() throws IOException {
		String path = "iris.csv";
		final File file= new File(path);
		Path ppath = FileSystems.getDefault().getPath(path);
		
		try {
			TestSuite.IRIS_DATASET.toFlatFile(true, file);
		}finally{
			Files.delete(ppath);
		}
	}
	
	@Test
	public void testWrite2() throws IOException {
		String path = "iris.csv";
		final File file= new File(path);
		Path ppath = FileSystems.getDefault().getPath(path);
		
		try {
			TestSuite.IRIS_DATASET.toFlatFile(false, file);
		}finally{
			Files.delete(ppath);
		}
	}
	
	@Test
	public void testWrite3() throws IOException {
		String path = "iris.csv";
		final File file= new File(path);
		Path ppath = FileSystems.getDefault().getPath(path);
		
		try {
			TestSuite.IRIS_DATASET.toFlatFile(true, file, '|');
		}finally{
			Files.delete(ppath);
		}
	}
	
	@Test
	public void testWrite4() throws IOException {
		String path = "iris.csv";
		final File file= new File(path);
		Path ppath = FileSystems.getDefault().getPath(path);
		
		try {
			TestSuite.IRIS_DATASET.toFlatFile(false, file, '|');
		}finally{
			Files.delete(ppath);
		}
	}
	
	@Test
	public void testGetSet() {
		DataSet irisCopy = TestSuite.IRIS_DATASET.copy();
		assertTrue(irisCopy.getEntry(0, 0) == 5.1);
		assertTrue(irisCopy.setEntry(0, 0, 6.5) == 5.1);
		assertTrue(irisCopy.getEntry(0, 0) == 6.5);
		assertTrue(TestSuite.IRIS_DATASET.getEntry(0, 0) == 5.1);
	}
	
	@Test
	public void testConstructorOneAndTwoAndNullData() {
		DataSet d = new DataSet(new double[][]{
			new double[]{0,1},
			new double[]{2,3}
		});
		
		assertNotNull(d.getHeaders());
		assertNull(d.getLabels());
		
		d = new DataSet(
			new Array2DRowRealMatrix(new double[][]{
				new double[]{0,1},
				new double[]{2,3}
		}));
		
		assertNotNull(d.getHeaders());
		assertNull(d.getLabels());
		
		boolean a = false;
		try {
			new DataSet(null, null, null, null, true);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		// test log in log4j
		KMeans k = new KMeansParameters(3).setVerbose(true).fitNewModel(TestSuite.IRIS_DATASET.getData());
		d.log(k);
	}
	
	@Test
	public void testBadHead() {
		boolean a = false;
		try {
			TestSuite.IRIS_DATASET.head(0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
}
