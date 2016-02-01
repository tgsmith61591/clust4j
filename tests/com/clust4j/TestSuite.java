package com.clust4j;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

import com.clust4j.algo.ClustTests;
import com.clust4j.algo.HDBSCANTests;
import com.clust4j.algo.HaversineTest;
import com.clust4j.algo.HeapTesting;
import com.clust4j.algo.HierTests;
import com.clust4j.algo.KMedoidsProtectedTests;
import com.clust4j.algo.SerializationTests;
import com.clust4j.algo.pipeline.PipelineTest;
import com.clust4j.algo.preprocess.ImputationTests;
import com.clust4j.algo.preprocess.PreProcessorTests;
import com.clust4j.data.TestDataSet;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.log.LogTest;
import com.clust4j.metrics.TestMetrics;
import com.clust4j.sample.BootstrapTest;
import com.clust4j.utils.BinarySearchTreeTests;
import com.clust4j.utils.MatTests;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.NNHSTests;
import com.clust4j.utils.TestDistanceEnums;
import com.clust4j.utils.TestUtils;
import com.clust4j.utils.VectorTests;
import com.clust4j.utils.parallel.ParallelTests;

@RunWith(Suite.class)
@Suite.SuiteClasses({
	ClustTests.class,
	KMedoidsProtectedTests.class,
	KernelTestCases.class,
	LogTest.class,
	BinarySearchTreeTests.class,
	MatTests.class,
	TestDistanceEnums.class,
	TestUtils.class,
	VectorTests.class,
	HaversineTest.class,
	SerializationTests.class,
	HeapTesting.class,
	HierTests.class,
	ImputationTests.class,
	BootstrapTest.class,
	PreProcessorTests.class,
	PipelineTest.class,
	ParallelTests.class,
	HDBSCANTests.class,
	NNHSTests.class,
	TestGlobals.class,
	TestDataSet.class,
	TestMetrics.class
})

/* Runs all the tests */
public class TestSuite {
	// Easy access to a global formatter for test classes
	public static final MatrixFormatter formatter = new MatrixFormatter();
}
