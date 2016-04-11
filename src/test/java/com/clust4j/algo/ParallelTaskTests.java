package com.clust4j.algo;

import org.junit.Test;

import com.clust4j.algo.ParallelChunkingTask.ChunkingStrategy;
import com.clust4j.algo.ParallelChunkingTask.SimpleChunkingStrategy;
import com.clust4j.algo.ParallelChunkingTask.CoreRestrictiveChunkingStrategy;
import com.clust4j.utils.MatUtils;

public class ParallelTaskTests {

	@Test
	public void test() {
		double[][] X = MatUtils.randomGaussian(750, 2);
		ChunkingStrategy strat = new SimpleChunkingStrategy();
		strat.map(X); // want to make sure it works.
		
		strat = new CoreRestrictiveChunkingStrategy(1);
		strat.map(X);
	}

}
