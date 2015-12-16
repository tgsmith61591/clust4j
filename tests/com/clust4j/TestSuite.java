package com.clust4j;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

import com.clust4j.algo.ClustTests;
import com.clust4j.algo.HaversineTest;
import com.clust4j.algo.HeapTesting;
import com.clust4j.algo.HierTests;
import com.clust4j.algo.KMedoidsProtectedTests;
import com.clust4j.algo.SerializationTests;
import com.clust4j.algo.prep.ImputationTests;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.log.LogTest;
import com.clust4j.utils.BinarySearchTreeTests;
import com.clust4j.utils.MatTests;
import com.clust4j.utils.TestDistanceEnums;
import com.clust4j.utils.TestUtils;
import com.clust4j.utils.VectorTests;

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
	ImputationTests.class
})

public class TestSuite {/* Runs all the tests */}
