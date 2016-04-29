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

package com.clust4j.algo;

import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.NamedEntity;
import com.clust4j.log.LogTimer;
import com.clust4j.utils.MatrixFormatter;

public abstract class ParallelChunkingTask<T> extends RecursiveTask<T> implements NamedEntity {
	private static final long serialVersionUID = 6377106189203872639L;
	final LogTimer timer;
	final public ChunkingStrategy strategy;
	final public ArrayList<Chunk> chunks;
	
	
	/**
	 * Wrapper for data chunks
	 * @author Taylor G Smith
	 */
	public static class Chunk implements java.io.Serializable {
		private static final long serialVersionUID = -4981036399670388292L;
		final double[][] chunk;
		final public int start;
		
		public Chunk(final double[][] c, int start_idx) {
			this.chunk = c;
			this.start = start_idx;
		}
		
		public double[][] get() { return chunk; }
		public int size() { return chunk.length; }
		@Override public String toString() { return new MatrixFormatter().format(chunk).toString(); }
	}
	
	
	/**
	 * The strategy for chunking the data
	 * @author Taylor G Smith
	 */
	abstract public static class ChunkingStrategy {
		public final static int AVAILABLE_CORES = GlobalState.ParallelismConf.NUM_CORES;
		public final static int DEF_CHUNK_SIZE = 500;
		final int chunkSize;
		
		public ChunkingStrategy() {
			this(DEF_CHUNK_SIZE);
		}
		
		public ChunkingStrategy(int chunkSize){
			this.chunkSize = chunkSize;
		}
		
		protected static Chunk getChunk(double[][] X, int chunkSize, int chunkNum) {
			double[][] chunk;
			
			int idx = 0;
			int startingPt = chunkNum * chunkSize;
			int endingPt = FastMath.min(X.length, startingPt + chunkSize);
			
			chunk = new double[endingPt - startingPt][];
			for(int j = startingPt; j < endingPt; j++) {
				chunk[idx++] = X[j];
			}
			
			return new Chunk(chunk, startingPt);
		}
		
		public int getChunkSize() {
			return chunkSize;
		}
		
		public static int getChunkSize(final int numRows) {
			return AVAILABLE_CORES == 1 ? numRows :
				FastMath.min(numRows, DEF_CHUNK_SIZE);
		}
		
		public int getNumChunks(final double[][] X) {
			if(1 == AVAILABLE_CORES)
				return 1;
			else {
				final int m = X.length;
				return getNumChunks(getChunkSize(), m);
			}
		}
		
		public static int getNumChunks(final int chunkSize, final int m) {
			return (int)FastMath.ceil( ((double)m)/((double)chunkSize) );
		}
		
		
		protected abstract ArrayList<Chunk> map(double[][] X);
	}
	
	/**
	 * Default chunking class
	 * @author Taylor G Smith
	 */
	static public class SimpleChunkingStrategy extends ChunkingStrategy {
		public SimpleChunkingStrategy() { 
			super(); 
		}
		
		@Override
		protected ArrayList<Chunk> map(double[][] X) {
			final ArrayList<Chunk> out = new ArrayList<>();
			final int numChunks = getNumChunks(X);
			
			for(int i = 0; i < numChunks; i++)
				out.add(getChunk(X, chunkSize, i));
			
			return out;
		}
	}
 	
	
	/**
	 * Default constructor
	 * @param X
	 */
	public ParallelChunkingTask(final double[][] X) {
		this(X, new SimpleChunkingStrategy());
	}
	
	/**
	 * Builds an instance with a default chunking strategy
	 * @param X
	 * @param strategy
	 */
	public ParallelChunkingTask(final double[][] X, final ChunkingStrategy strategy) {
		this.timer = new LogTimer();
		this.strategy = strategy;
		this.chunks = strategy.map(X);
	}
	
	/**
	 * Builds an instance with chunks already generated
	 * @param chunks
	 * @param strategy
	 */
	public ParallelChunkingTask(ParallelChunkingTask<T> task) {
		this.timer = new LogTimer();
		this.strategy = task.strategy;
		this.chunks = task.chunks;
	}

	

	@Override
	public String getName() {
		return formatName(Thread.currentThread().getName());
	}
	
	public static ForkJoinPool getThreadPool() {
		return GlobalState.ParallelismConf.FJ_THREADPOOL;
	}
	
	public String formatName(String str) {
		StringBuilder sb = new StringBuilder();
		boolean hyphen = false; // have we hit the hyphen yet?
		
		for(char c: str.toCharArray()) {
			if(hyphen || Character.isUpperCase(c))
				sb.append(c);
			
			else if('-' == c) {
				hyphen = true;
				sb.append(c);
			}
		}
		
		return sb.toString();
	}
	
	/**
	 * The operation to perform on each chunk.
	 * @param chunk
	 * @return
	 */
	public abstract T reduce(Chunk chunk);
}
