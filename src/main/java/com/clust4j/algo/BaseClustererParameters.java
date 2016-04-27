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

import java.util.Random;

import com.clust4j.Clust4j;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.DeepCloneable;

/**
 * Base planner class many clustering algorithms
 * will extend with static inner classes. Some clustering
 * algorithms will require more parameters and must provide
 * the interface for the getting/setting of such parameters.
 * 
 * @author Taylor G Smith
 */
abstract public class BaseClustererParameters 
		extends Clust4j // So all are serializable
		implements DeepCloneable, BaseClassifierParameters {
	private static final long serialVersionUID = -5830795881133834268L;
	
	protected boolean parallel, 
		verbose = AbstractClusterer.DEF_VERBOSE;
	protected Random seed = AbstractClusterer.DEF_SEED;
	protected GeometricallySeparable metric = AbstractClusterer.DEF_DIST;
	
	@Override abstract public BaseClustererParameters copy();
	abstract public BaseClustererParameters setSeed(final Random rand);
	abstract public BaseClustererParameters setVerbose(final boolean b);
	abstract public BaseClustererParameters setMetric(final GeometricallySeparable dist);
	abstract public BaseClustererParameters setForceParallel(final boolean b);

	final public GeometricallySeparable getMetric() { return metric; }
	final public boolean getParallel() 				{ return parallel; }
	final public Random getSeed() 					{ return seed; }
	final public boolean getVerbose() 				{ return verbose; }
}
