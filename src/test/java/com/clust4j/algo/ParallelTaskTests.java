package com.clust4j.algo;

import static org.junit.Assert.*;

import org.junit.Test;

import com.clust4j.algo.ParallelChunkingTask.ChunkingStrategy;
import com.clust4j.algo.ParallelChunkingTask.SimpleChunkingStrategy;
import com.clust4j.utils.MatUtils;

public class ParallelTaskTests {

	@Test
	public void test() {
		double[][] X = MatUtils.randomGaussian(750, 2);
		ChunkingStrategy strat = new SimpleChunkingStrategy();
		strat.map(X); // want to make sure it works.
		
		// make sure works, no NPEs
		assertNotNull(strat.map(X).get(0).toString());
		assertTrue(strat.getNumChunks(X) > 0);
		
		// there's a format name method for fork join pool tasks..
		assertNotNull(new ParallelChunkingTask<Integer>(X){
			private static final long serialVersionUID = 1L;
			@Override public Integer reduce(com.clust4j.algo.ParallelChunkingTask.Chunk chunk){ return -1; }
			@Override protected Integer compute() { return -1; }
		}.formatName("FJ-1-1"));
	}

}
