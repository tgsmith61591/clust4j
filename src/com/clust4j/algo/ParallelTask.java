package com.clust4j.algo;

import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.log.LogTimer;
import com.clust4j.utils.Named;

abstract class ParallelTask<T> extends RecursiveTask<T> implements Named {
	private static final long serialVersionUID = 6377106189203872639L;
	final LogTimer timer;
	
	interface Chunker<T> {
		T doChunk(double[][] chunk);
	}
	
	ParallelTask() {
		this.timer = new LogTimer();
	}
	
	/**
	 * Build the concurrent stack of chunks...
	 * @param X
	 * @return
	 */
	static ArrayList<double[][]> generateChunks(double[][] X, int numChunks) {
		final int chunkSize = X.length / numChunks;
		
		ArrayList<double[][]> output = new ArrayList<>();
		for(int i = 0; i < numChunks; i++)
			output.add(getChunk(X, chunkSize, i));
		
		return output;
	}
	
	static double[][] getChunk(double[][] X, int chunkSize, int num) {
		double[][] chunk;
		
		int idx = 0;
		int startingPt = num * chunkSize;
		int endingPt = FastMath.min(X.length, startingPt + chunkSize); 
		
		chunk = new double[endingPt - startingPt][];
		for(int j = startingPt; j < endingPt; j++)
			chunk[idx++] = X[j];
		
		return chunk;
	}

	@Override
	public String getName() {
		return formatName(Thread.currentThread().getName());
	}
	
	public static ForkJoinPool getThreadPool() {
		return GlobalState.ParallelismConf.FJ_THREADPOOL;
	}
	
	private static String formatName(String str) {
		return str.replace("ForkJoinPool", "FJ")
			.replace("worker", "task");
	}
}
