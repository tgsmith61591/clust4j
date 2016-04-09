package com.clust4j;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
	TestSuite.class,
	LoadTests.class
})

/**
 * Secondary test suite for clust4j. Runs all production
 * tests as well as some larger tests we don't want TravisCI to run
 * @author Taylor G Smith
 */
public class TestAll {
}
