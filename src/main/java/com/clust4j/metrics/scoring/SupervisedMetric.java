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
package com.clust4j.metrics.scoring;

import java.util.HashSet;
import java.util.TreeMap;

import org.apache.commons.math3.exception.DimensionMismatchException;

public enum SupervisedMetric implements EvaluationMetric {
	BINOMIAL_ACCURACY {
		@Override
		public double evaluate(final int[] actual, final int[] predicted) {
			return numEqual(actual, predicted) / (double)actual.length;
		}
	},

	/**
	 * The issue we face in multiclass scoring for unsupervised learning
	 * algorithms is that depending on the random state of the model, we
	 * may classify actual labels of <tt>{0,1,0,2,2}</tt> into <tt>{2,0,2,1,1}</tt>
	 * and in a sense, we are completely accurate in terms of segmentation and 
	 * purity... but we need an accurate way to measure this. Traditional information 
	 * retrieval definition of accuracy (IRA) will actually score this situation as 0% accurate, 
	 * even though its identified the perfect class separation.
	 * 
	 * <p>
	 * This method, then, is an attempt to measure accuracy not traditionally, but by
	 * accounting for predicted segmentation in regards to actual label segmentation. It works
	 * by penalizing indices which are inappropriately associated with incorrect neighbor indices.
	 * For instance&mdash;in the above example&mdash;there are no infractions counted, as index
	 * <tt>0</tt> is correctly associated with index <tt>2</tt>, etc. However for the example where:
	 * 
	 * <p>
	 * <tt>actual = {0,1,0,2,2}</tt><br>
	 * <tt>predicted = {2,0,1,2,1}</tt>
	 * <p>
	 * 
	 * ...the accuracy will actually be computed as 0.6, as opposed to the IRA score of 0.2.
	 * Therefore, this method works robustly where traditional accuracy scoring or even the 
	 * use of a {@link LabelEncoder} will not (due to potentially inconsistent ordering).
	 * 
	 * @param actual - an <tt>int[]</tt> of the true labels
	 * @param predicted - an <tt>int[]</tt> of the predicted labels
	 * @throws DimensionMismatchException if the dimensions of actual and predicted to not match
	 * @throws IllegalArgumentException if the number of classes in <tt>actual</tt> does not match
	 * the number of classes in <tt>predicted</tt>
	 * @return an accuracy score measuring class segmentation in actual vs. predicted, 
	 * and whether the proper class boundaries (class label agnostic) were identified
	 * @author Taylor G Smith
	 */
	INDEX_AFFINITY {
		@Override
		public double evaluate(int[] actual, int[] predicted) {
		
			// Ensure equal dims
			final int n = actual.length;
			if(n != predicted.length)
				throw new DimensionMismatchException(n, predicted.length);
			if(0 == n)
				throw new IllegalArgumentException("cannot score empty labels");
		
			// Generate trees and counts for each array's class and counts
			TreeMap<Integer, Integer> actualCounts    = new TreeMap<>();
			TreeMap<Integer, Integer> predictedCounts = new TreeMap<>();
		
			// Simultaneously generate the trees and counts
			Integer actLab, predLab, actVal, predVal;
		
			for(int i = 0; i < n; i++) {
				actLab = actual[i];
				predLab= predicted[i];
		
				actVal = actualCounts.get(actLab);
				predVal= predictedCounts.get(predLab);
		
				// Put or increment act
				if(null == actVal) actualCounts.put(actLab, 1);
				else actualCounts.put(actLab, actVal + 1); // Avoid another get operation
			
				// Put or increment pred
				if(null == predVal) predictedCounts.put(predLab, 1);
				else predictedCounts.put(predLab, predVal + 1); // Avoid another get operation
			}
		
			final int numActLabels = actualCounts.size();
			final int numPredLabels = predictedCounts.size();
		
			/*
			 * Our easiest case is that there are equal numbers of classes in each...
			 * this is our base case and perhaps the easiest one to solve for, so we should
			 * handle it first.
			 */
		
			if( numActLabels == numPredLabels ) {
		
				/*
				 * Base case within a base case: what if they are both length 1?
				 */
			
				if( 1 == numActLabels )
					return 1.0;
			
				/*
				 * Second, and probably more rare, is that there are separate labels for each
				 * record. This one can occur in k-based models or in agglomerative models...
				 */
			
				if(numActLabels == n)
					return 1.0;
			
				/*
				 * Otherwise, and more likely, we have a situation like:
				 * 
				 * actual    = {0,1,0,2,2};
				 * predicted = {2,0,2,1,1};
				 * 
				 * In this case, we need to look at indices grouped together
				 * for each set of labels. This standardizes our computation:
				 * 
				 * actual    = {[0,2], [1], [3,4]}
				 * predicted = {[0,2], [1], [3,4]}
				 *
				 * Thus, the computation becomes more a Levenshtein distance
				 * computation across arrays of indices. However, things get more
				 * complicated in a situation like this:
				 * 
				 * actual    = {0,1,0,2,2};
				 * predicted = {2,0,1,2,1};
				 * 
				 * ...where the indices look like:
				 * 
				 * actual    = {[0,2], [1], [3,4]}
				 * predicted = {[0,3], [1], [2,4]}
				 * 
				 * ... in this case, though it looks uglier, there are actually only two
				 * indices (2,3) in the predicted labels that are mis-associated with peers.
				 * 
				 * Logically it should read:
				 *   - For each index, identify the peer set it's associated with (pred and act)
				 *   - Compute the number of discrepancies such that (if idx = 0) [2] vs. [3] is one discrepancy
				 *     and [2] vs. [2,3] is also one discrepancy, but [2] vs. [1,3] is two. There should
				 *     not be a situation where the number of discrepancies exceeds the number of indices (n).
				 *   - The accuracy will be computed as 1 - (numDiscrepancies / n)
				 *
				 * This makes the total runtime for this case O(Nchoose2 * N * p), where p is the lookup
				 * time for the HashSets. For a smaller label set, p should be nearly negligible.
				 */
			
				// First, identify for each index in range(n)--in both act and pred--make a map of
				// <Integer, HashSet<Integer>> this, unfortunately, is an O(N choose 2) operation
				TreeMap<Integer, HashSet<Integer>> actAssns  = new TreeMap<>();
				TreeMap<Integer, HashSet<Integer>> predAssns = new TreeMap<>();
			
				// Init
				int i;
				actAssns.put (0, new HashSet<Integer>());
				predAssns.put(0, new HashSet<Integer>());
			
				for(i = 0; i < n - 1; i++) {
					actLab = actual[i];
					predLab= predicted[i];
				
					for(int j = i + 1; j < n; j++) {
				
						// These hold index : indices with the same label
						// Only need to do this on the first pass
					
						if(0 == i) {
							actAssns.put (j, new HashSet<Integer>());
							predAssns.put(j, new HashSet<Integer>());
						}
					
						// associated with this index
						if(actLab == actual[j]) {
							actAssns.get(i).add(j);
							actAssns.get(j).add(i);
						}
					
						// associated with this index
						if(predLab == predicted[j]) {
							predAssns.get(i).add(j);
							predAssns.get(j).add(i);
						}
					}
				}
			
				// Now the assn trees have been mapped, time to go to step 2 counting infractions
				// We can't penalize for presence AND absence from clusters, or we will
				// end up double counting. Thus, we should only count an infraction if present
				// and should not be present.
			
				HashSet<Integer> truth, pred;
				HashSet<Integer> violatingIdcs = new HashSet<Integer>();
				int infractions = 0;
			
				for(i = 0; i < n; i++) {
			
					truth = actAssns.get(i);
					pred  = predAssns.get(i);
				
					pred.removeAll(truth);
					pred.removeAll(violatingIdcs);
				
					for(Integer p: pred) {
						//System.out.println(p);
						violatingIdcs.add(p);
						infractions++;
					}
				
					// Already have a link between the bad idcs and these. Don't want to double count
					violatingIdcs.addAll(truth);
				}
			
				return 1.0 - ((double)infractions / (double)n);
			}
			
			/* 
			 * We have a difficult situation if the number of classes in actual
			 * differs from the number of classes in predicted... i.e., if the model
			 * is not k-based (or even if it is and the user selects a strange k value)
			 * and identifies clusters via density-based means or otherwise. For now,
			 * let's not support this corner case and we can address it later...
			 */
		
			throw new IllegalArgumentException("num predicted classes != "
				+ "num actual classes (" + numActLabels +", " + numPredLabels + ")");
		}
	},
	
	// TODO: more?
	;
	
	private static void checkDims(int[] a, int[] b) {
		if(a.length != b.length) // Allow empty; so we don't use VecUtils
			throw new DimensionMismatchException(a.length, b.length);
	}
	
	private static int numEqual(int[] a, int[] b) {
		checkDims(a, b);
		int sum = 0;
		for(int i = 0; i < a.length; i++)
			if(a[i] == b[i])
				sum++;
		return sum;
	}
	
	abstract public double evaluate(final int[] actual, final int[] predicted);
	// TODO: tp/fp/tn/fn for multiclass...
}
