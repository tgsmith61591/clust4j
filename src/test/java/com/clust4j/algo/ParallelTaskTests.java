package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;

import com.clust4j.algo.ParallelChunkingTask.Chunk;
import com.clust4j.algo.ParallelChunkingTask.ChunkingStrategy;
import com.clust4j.algo.ParallelChunkingTask.SimpleChunkingStrategy;
import com.clust4j.utils.MatUtils;

public class ParallelTaskTests {

	@Test
	public void test() {
		double[][] X = MatUtils.randomGaussian(750, 2);
		ChunkingStrategy strat = new SimpleChunkingStrategy();
		ArrayList<Chunk> chunks = strat.map(X);
		assertTrue(chunks.size() == 2);
		assertTrue(strat.getNumChunks(X) == 2);
		assertTrue(ChunkingStrategy.getChunk(X, 500, 1).size() == 250);
	}

}
