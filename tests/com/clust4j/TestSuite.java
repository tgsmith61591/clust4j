package com.clust4j;

import java.io.File;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

import com.clust4j.algo.AffinityPropagationTests;
import com.clust4j.algo.ClustTests;
import com.clust4j.algo.DBSCANTests;
import com.clust4j.algo.HDBSCANTests;
import com.clust4j.algo.HeapTesting;
import com.clust4j.algo.HierarchicalTests;
import com.clust4j.algo.KMeansTests;
import com.clust4j.algo.KMedoidsTests;
import com.clust4j.algo.MeanShiftTests;
import com.clust4j.algo.NNHSTests;
import com.clust4j.algo.NearestCentroidTests;
import com.clust4j.algo.NearestNeighborsTests;
import com.clust4j.algo.TestLabelEncoder;
import com.clust4j.algo.RadiusNeighborsTests;
import com.clust4j.algo.pipeline.PipelineTest;
import com.clust4j.algo.preprocess.ImputationTests;
import com.clust4j.algo.preprocess.PreProcessorTests;
import com.clust4j.data.TestDataSet;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.log.LogTest;
import com.clust4j.metrics.pairwise.HaversineTest;
import com.clust4j.metrics.pairwise.PairwiseTests;
import com.clust4j.metrics.pairwise.TestDistanceEnums;
import com.clust4j.metrics.scoring.TestMetrics;
import com.clust4j.sample.BootstrapTest;
import com.clust4j.utils.BinarySearchTreeTests;
import com.clust4j.utils.FormatterTests;
import com.clust4j.utils.MatTests;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.QuadTupTests;
import com.clust4j.utils.SeriesTests;
import com.clust4j.utils.TestUtils;
import com.clust4j.utils.VectorTests;
import com.clust4j.utils.parallel.ParallelTests;

@RunWith(Suite.class)
@Suite.SuiteClasses({
	AffinityPropagationTests.class,
	BinarySearchTreeTests.class,
	BootstrapTest.class,
	ClustTests.class,
	DBSCANTests.class,
	FormatterTests.class,
	HaversineTest.class,
	HDBSCANTests.class,
	HeapTesting.class,
	HierarchicalTests.class,
	ImputationTests.class,
	KMeansTests.class,
	KMedoidsTests.class,
	KernelTestCases.class,
	LogTest.class,
	MatTests.class,
	MeanShiftTests.class,
	NearestCentroidTests.class,
	NearestNeighborsTests.class,
	NNHSTests.class,
	PairwiseTests.class,
	ParallelTests.class,
	PipelineTest.class,
	PreProcessorTests.class,
	QuadTupTests.class,
	RadiusNeighborsTests.class,
	SeriesTests.class,
	TestDataSet.class,
	TestDistanceEnums.class,
	TestGlobals.class,
	TestLabelEncoder.class,
	TestMetrics.class,
	TestUtils.class,
	VectorTests.class
})

/* Runs all the tests */
public class TestSuite {
	// Easy access to a global formatter for test classes
	public static final MatrixFormatter formatter = new MatrixFormatter();
	public static String tmpSerPath = "model.ser";
	public static File file = new File(tmpSerPath);
	public static Path path = FileSystems.getDefault().getPath(tmpSerPath);
	
	
	public static final double[][] bigMatrix = new double[][]{
		new double[]{ 0.08594657,  0.60925865,  0.39881186,  0.26410921,  0.19803359, 0.21035565,  0.11966197,  0.52581139,  0.38387628,  0.4825036 },
		new double[]{ 0.56943745,  0.9438055 ,  0.03867595,  0.49143331,  0.27470736, 0.08862225,  0.6203588 ,  0.76004573,  0.28224907,  0.98504973},
		new double[]{ 0.83903022,  0.48852263,  0.06229877,  0.00903001,  0.63412978, 0.01088595,  0.34147105,  0.31239485,  0.54709824,  0.60489573},
		new double[]{ 0.73606295,  0.64216898,  0.04652937,  0.00361512,  0.26470677, 0.59203757,  0.6751856 ,  0.36033876,  0.91600272,  0.97278172},
		new double[]{ 0.58265159,  0.82568491,  0.18918761,  0.85330221,  0.12915527, 0.21483348,  0.4457545 ,  0.61225749,  0.59133551,  0.74850421},
		new double[]{ 0.57694678,  0.71203528,  0.93801156,  0.12880288,  0.37977797, 0.1621018 ,  0.28997765,  0.48443738,  0.22086107,  0.33982515},
		new double[]{ 0.44102127,  0.03442744,  0.85580514,  0.49294382,  0.49326928, 0.16050842,  0.07405442,  0.55629147,  0.24660354,  0.12663662},
		new double[]{ 0.23211021,  0.94979067,  0.05108022,  0.59183824,  0.14980919, 0.32508404,  0.59326028,  0.92051835,  0.05639324,  0.53309613},
		new double[]{ 0.04267053,  0.78979235,  0.67823901,  0.47411163,  0.47987525, 0.64574195,  0.19336916,  0.75687751,  0.62494332,  0.31120583},
		new double[]{ 0.35446975,  0.90690062,  0.84454885,  0.37504377,  0.30435096, 0.91211773,  0.00132654,  0.08953336,  0.77461863,  0.51186425}
	};
	
	public static Array2DRowRealMatrix getRandom(final int rows, final int cols) {
		final Random rand = new Random();
		final double[][] data = new double[rows][cols];
		
		for(int i = 0; i < rows; i++)
			for(int j = 0; j < cols; j++)
				data[i][j] = rand.nextDouble() * (rand.nextDouble() > 0.5 ? -1 : 1);
		
		return new Array2DRowRealMatrix(data, false);
	}
}
