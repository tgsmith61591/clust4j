package com.clust4j.utils;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public class GridSearchUtils {
	/**
	 * A class that splits an AbstractRealMatrix into two or three parts
	 * @author Taylor G Smith
	 */
	abstract static class SplitFactory implements TrainTestSplit {
		final public static Random DEF_SEED = new Random();
		final protected AbstractRealMatrix data;
		final protected Random seed;
		
		public SplitFactory(final AbstractRealMatrix data) { this(data, DEF_SEED); }
		public SplitFactory(final AbstractRealMatrix data, final Random seed) { 
			this.data = (AbstractRealMatrix) data.copy();
			this.seed = seed; 
		}
		
		abstract protected void defineSplit();
	}
	
	
	/**
	 * A class that splits an AbstractRealMatrix into two parts
	 * @author Taylor G Smith
	 */
	public static class TrainTestSplitFactory extends SplitFactory {
		public TrainTestSplitFactory(final AbstractRealMatrix data) {
			super(data);
		}
		
		public TrainTestSplitFactory(final AbstractRealMatrix data, final Random seed) {
			super(data, seed);
		}

		@Override
		protected void defineSplit() {
			// TODO Auto-generated method stub
		}

		@Override
		public AbstractRealMatrix trainSet() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public AbstractRealMatrix testSet() {
			// TODO Auto-generated method stub
			return null;
		}
	}
	
	/**
	 * A class that splits an AbstractRealMatrix into three parts
	 * @author Taylor G Smith
	 */
	public static class TrainTestHoldoutSplitFactory extends SplitFactory implements TrainTestHoldoutSplit {
		public TrainTestHoldoutSplitFactory(final AbstractRealMatrix data) {
			super(data);
		}
		
		public TrainTestHoldoutSplitFactory(final AbstractRealMatrix data, final Random seed) {
			super(data, seed);
		}

		@Override
		protected void defineSplit() {
			// TODO Auto-generated method stub
			
		}

		@Override
		public AbstractRealMatrix holdoutSet() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public AbstractRealMatrix trainSet() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public AbstractRealMatrix testSet() {
			// TODO Auto-generated method stub
			return null;
		}
	}
	
	
}
