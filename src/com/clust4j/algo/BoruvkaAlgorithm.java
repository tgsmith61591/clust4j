package com.clust4j.algo;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.algo.NearestNeighborHeapSearch.NodeData;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Loggable;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.DistanceMetric;
import com.clust4j.utils.VecUtils;

class BoruvkaAlgorithm implements java.io.Serializable {
	private static final long serialVersionUID = 3935595821188876442L;

	// the initialization reorganizes the trees
	final BoruvAlg alg;
	
	private final NearestNeighborHeapSearch outer_tree;
	private final int minSamples;
	private final DistanceMetric metric;
	private final boolean approxMinSpanTree;
	private final int leafSize;
	private final Loggable logger;
	private final double alpha;
	
	BoruvkaAlgorithm(NearestNeighborHeapSearch tree, int min_samples, 
			DistanceMetric metric, int leafSize, boolean approx_min_span_tree,
			double alpha) {
		this(tree, min_samples, metric, leafSize, approx_min_span_tree, alpha, null);
	}
	
	BoruvkaAlgorithm(NearestNeighborHeapSearch tree, int min_samples, 
			DistanceMetric metric, int leafSize, boolean approx_min_span_tree,
			double alpha, Loggable logger) {
		
		this.outer_tree = tree;
		this.minSamples = min_samples;
		this.metric = metric;
		this.leafSize = leafSize;
		this.approxMinSpanTree = approx_min_span_tree;
		this.alpha = alpha;
		this.logger = logger;
		
		
		// Create the actual solver -- if using logger,
		// updates with info in the actual algorithm
		alg = (tree instanceof KDTree) ? 
				new KDTreeBoruvAlg() :
					new BallTreeBoruvAlg();
	}
	
	
	/*
	static class BoruvkaUnionFind extends HDBSCAN.UnifiedFinder {
		int[][] data;
		boolean[] isComponent;
		
		BoruvkaUnionFind(int size) {
			super(size);
			data = new int[size][2];
			
			// Set first col to arange
			for(int i = 0; i < size; i++)
				data[i][0] = i;
			
			isComponent = new boolean[size];
		}
		
		@Override
		public void union(int x, int y) {
			int xRoot = find(x);
			int yRoot = find(y);
			
			if(data[xRoot][1] < data[yRoot][1])
				data[xRoot][0] = yRoot;
			else if(data[xRoot][1] > data[yRoot][1])
				data[yRoot][0] = xRoot;
			else {
				data[yRoot][0] = xRoot;
				data[xRoot][1] += 1;
			}
		}
		
		@Override
		public int find(int x) {
			if(data[x][0] != x) {
				data[x][0] = find(data[x][0]);
				isComponent[x] = false;
			}
			
			return data[x][0];
		}
		
		int[] components() {
			ArrayList<Integer> nonZero = new ArrayList<>();
			
			for(int i = 0; i < isComponent.length; i++)
				if(isComponent[i])
					nonZero.add(i);
			
			int[] components = new int[nonZero.size()];
			for(int i = 0; i < components.length; i++)
				components[i] = nonZero.get(i);
			
			return components;
		}
	}
	*/
	
	static class BoruvkaUnionFind extends HDBSCAN.TreeUnionFind {
		BoruvkaUnionFind(int N) {
			super(N);
		}
	}
	

	double ballTreeMinDistDual(double rad1, double rad2, int node1, int node2, double[][] centroidDist) {
		double distPt = centroidDist[node1][node2];
		return FastMath.max(0, (distPt - rad1 - rad2));
	}
	
	Neighborhood coreDistQuery(NearestNeighborHeapSearch tree, double[][] X,
									int min_samples, boolean dualTree, boolean breadthFirst) {
		return tree.query(X, min_samples, dualTree, true);
	}
	
	/**
	 * Similar to {@link KDTree}<tt>.minRDistDual(...)</tt> but
	 * uses one node bounds array instead of two instances of
	 * {@link NearestNeighborHeapSearch}
	 * @param metric
	 * @param node1
	 * @param node2
	 * @param nodeBounds
	 * @param n
	 * @return
	 */
	double kdTreeMinDistDual(DistanceMetric metric, int node1, int node2, double[][][] nodeBounds, int n) {
		return metric.partialDistanceToDistance(kdTreeMinRDistDual(metric, node1, node2, nodeBounds, n));
	}
	
	double kdTreeMinRDistDual(DistanceMetric metric, int node1, int node2, double[][][] nodeBounds, int n) {
		double d, d1, d2, rdist = 0.0;
		int j;
		
		for(j = 0; j < n; j++) {
			d1 = nodeBounds[0][node1][j] - nodeBounds[1][node2][j];
			d2 = nodeBounds[0][node2][j] - nodeBounds[1][node1][j];
			d = (d1 + FastMath.abs(d1)) + (d2 + FastMath.abs(d2));
			
			if(metric.getP() == Double.POSITIVE_INFINITY)
				rdist = FastMath.max(rdist, 0.5 * d);
			else
				rdist += FastMath.pow(0.5 * d, metric.getP());
		}
		
		return rdist;
	}
	
	
	/**
	 * The {@link NearestNeighborHeapSearch} 
	 * tree traversal algorithm
	 * @author Taylor G Smith
	 */
	abstract class BoruvAlg {
		final static int INIT_VAL = -1;
		
		final NearestNeighborHeapSearch coreDistTree = outer_tree;
		final NearestNeighborHeapSearch TREE;
		final BoruvkaUnionFind componentUnionFind;
		
		final double[][] data_arr;
		final double[][][] node_bounds;
		final int[] idx_array;
		final NodeData[] node_data;
		final boolean partialDistTransform;
		
		int numPoints, numFeatures, 
			numNodes, numEdges;
		double[] bounds;
		int[] components, 
			  componentOfPoint, 
			  componentOfNode, 
			  candidateNeighbors, 
			  candidatePoint;
		double[] candidateDistance;
		double[][] edges;
		double[] coreDistance;
		
		BoruvAlg(boolean partialTrans, NearestNeighborHeapSearch TREE){
			this.TREE 			= TREE;
			this.data_arr 		= TREE.getDataRef();
			this.node_bounds 	= TREE.getNodeBoundsRef();
			this.idx_array 		= TREE.getIndexArrayRef();
			this.node_data 		= TREE.getNodeDataRef();

			this.numPoints 		= this.data_arr.length;
			this.numFeatures	= this.data_arr[0].length;
			this.numNodes 		= this.node_bounds.length;

			this.components 		= VecUtils.arange(numPoints);
			this.bounds 			= new double[numNodes];
			this.componentOfPoint 	= new int[numPoints];
			this.componentOfNode 	= new int[numNodes];
			this.candidateNeighbors	= new int[numPoints];
			this.candidatePoint 	= new int[numPoints];
			this.candidateDistance 	= new double[numPoints];
			this.edges 				= new double[numPoints - 1][3];
			this.componentUnionFind = new BoruvkaUnionFind(numPoints);
			
			long s = System.currentTimeMillis();
			this.partialDistTransform = partialTrans;
			
			initComponents();
			computeBounds();

			if(null != logger)
				logger.info("completed Boruvka nearest neighbor search in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-s, false) + 
					System.lineSeparator());
		}
		
		final void initComponents() {
			int n;
			
			for(n = 0; n < this.numPoints; n++) {
				this.componentOfPoint[n]	= n;
				this.candidateNeighbors[n]	= INIT_VAL;
				this.candidatePoint[n]		= INIT_VAL;
				this.candidateDistance[n]	= Double.MAX_VALUE;
			}
			
			for(n = 0; n < numNodes; n++)
				this.componentOfNode[n] = -(n + 1);
		}
		
		final double[][] spanningTree() {
			int numComponents = this.data_arr.length;
			
			while(numComponents > 1) {
				this.dualTreeTraversal(0, 0);
				numComponents = this.updateComponents();
			}
			
			return this.edges;
		}
		
		final int updateComponents() {
			int source, sink, c, component, n, i, p, currentComponent,
				currentSrcComponent, currentSinkComponent, child1, child2,
				lastNumComponents;
			NodeData nodeInfo;
			
			// For each component there should be a:
		    //   - candidate point (a point in the component)
		    //   - candidate neighbor (the point to join with)
		    //   - candidate_distance (the distance from point to neighbor)
		    //
		    // We will go through and and an edge to the edge list
		    // for each of these, and the union the two points
		    // together in the union find structure
			for(c = 0; c < this.components.length; c++ /* <- tee-hee */) {
				component = this.components[c];
				source = this.candidatePoint[component];
				sink = this.candidateNeighbors[component];
				
				if(source == INIT_VAL || sink == INIT_VAL)
					continue;
				
				currentSrcComponent = this.componentUnionFind.find(source);
				currentSinkComponent= this.componentUnionFind.find(sink);
				
				if(currentSrcComponent == currentSinkComponent) {
					// Already joined so ignore this edge
					this.candidatePoint[component] = INIT_VAL;
					this.candidateNeighbors[component] = INIT_VAL;
					this.candidateDistance[component] = Double.MAX_VALUE;
					continue;
				}
				
				this.edges[numEdges][0] = source;
				this.edges[numEdges][1] = sink;
				this.edges[numEdges][2] = this.partialDistTransform ?
						metric.partialDistanceToDistance(
							this.candidateDistance[component]) :
								this.candidateDistance[component];
				this.numEdges++;
				this.componentUnionFind.union(source, sink);
				
				// Reset and check for termination condition
				this.candidateDistance[component] = Double.MAX_VALUE;
				if(this.numEdges == this.numPoints - 1) {
					this.components = this.componentUnionFind.components();
					return components.length;
				}
			}
			
			
			// After joining everything, we go through to determine
			// the components of each point for an easier lookup.
			for(n = 0; n < data_arr.length; n++)
				this.componentOfPoint[n] = this.componentUnionFind.find(n);
			
			for(n = node_data.length - 1; n >= 0; n--) {
				nodeInfo = this.node_data[n];
				
				// If node is leaf, check that every point in node is same component
				if(nodeInfo.isLeaf()) {
					currentComponent = this.componentOfPoint[idx_array[nodeInfo.start()]];
					
					boolean found = false;
					for(i = nodeInfo.start() + 1; i < nodeInfo.end(); i++) {
						p = idx_array[i];
						if(componentOfPoint[p] != currentComponent) {
							found = true;
							break;
						}
					}
					
					// Alternative to the python for... else construct.
					if(!found)
						this.componentOfNode[n] = currentComponent;
				}
				
				// If not leaf, check both child nodes are same component
				else {
					child1 = 2 * n + 1;
					child2 = 2 * n + 2;
					
					if(this.componentOfNode[child1] == this.componentOfNode[child2])
						this.componentOfNode[n] = this.componentOfNode[child1];
				}
			}
			
			// This is a tie breaking method
			if(approxMinSpanTree) {
				lastNumComponents = this.components.length;
				components = this.componentUnionFind.components();
				
				if(components.length == lastNumComponents) // i.e., if all is isComponents are true
					for(n = 0; n < numNodes; n++) // Reset
						bounds[n] = Double.MAX_VALUE;
				
			} else {
				this.components = this.componentUnionFind.components();
				for(n = 0; n < numNodes; n++)
					this.bounds[n] = Double.MAX_VALUE;
			}
			
			return components.length;
		}

		abstract void computeBounds();
		abstract int dualTreeTraversal(int node1, int node2);
	}
	
	class KDTreeBoruvAlg extends BoruvAlg {
		KDTreeBoruvAlg() {
			super(true, new KDTree(
				new Array2DRowRealMatrix(outer_tree.getDataRef(), false), 
				leafSize, metric, logger));
		}
		
		@Override
		void computeBounds() {
			int n, i, m;
			
			Neighborhood queryResult =
				TREE.query(data_arr, minSamples + 1, true, true);
		
			double[][] knnDist = queryResult.getDistances();
			int[][] knnIndices = queryResult.getIndices();
			
			// Assign the core distance array and change to rdist...
			this.coreDistance = new double[knnDist.length];
			for(i = 0; i < coreDistance.length; i++)
				coreDistance[i] = metric
					.distanceToPartialDistance(
						knnDist[i][minSamples]);
			
			for(n = 0; n < numPoints; n++) {
				for(i = 1; i < minSamples + 1; i++) {
					m = knnIndices[n][i];
					
					if(this.coreDistance[m] <= this.coreDistance[n]) {
						this.candidatePoint[n] 	   = n;
						this.candidateNeighbors[n] = m;
						this.candidateDistance[n]  = this.coreDistance[n];
						break;
					}
				}
			}
			
			updateComponents();
			
			for(n = 0; n < numNodes; n++)
				this.bounds[n] = Double.MAX_VALUE;
		}

		@Override
		int dualTreeTraversal(int node1, int node2) {
			int[] pointIndices1, pointIndices2;
			int i, j, p, q, parent;
			
			double nodeDist, d, mrDist, newBound,
				newUpperBound, newLowerBound,
				leftDist, rightDist;
			
			NodeData node1Info = node_data[node1],
					 node2Info = node_data[node2];
			
			int component1, component2, left, right;
			
			// Distance btwn query and ref nodes
			nodeDist = kdTreeMinRDistDual(metric, node1, node2, 
					this.node_bounds, this.numFeatures);
			
			// If dist < current bound and nodes are not in the
			// same component, we continue
			if(nodeDist < this.bounds[node1]) {
				if(this.componentOfNode[node1] == this.componentOfNode[node2]
					&& this.componentOfNode[node1] >= 0)
					return 0;
				else {
					/*
					 * Pass. This is the only condition in which
					 * the method will continue without exiting early
					 */
				}
			} else
				return 0;
			
			
			
			// If both nodes are leaves
			if(node1Info.isLeaf() && node2Info.isLeaf()) {
				newUpperBound = 0.0;
				newLowerBound = Double.MAX_VALUE;
				
				// Build the indices
				pointIndices1 = new int[node1Info.end() - node1Info.start()];
				pointIndices2 = new int[node2Info.end() - node2Info.start()];
				
				// Populate the indices
				for(i = node1Info.start(), j = 0; i < node1Info.end(); i++, j++)
					pointIndices1[j] = this.idx_array[i];
				for(i = node2Info.start(), j = 0; i < node2Info.end(); i++, j++)
					pointIndices2[j] = this.idx_array[i];
				
				
				for(i = 0; i < pointIndices1.length; i++) {
					p = pointIndices1[i];
					component1 = this.componentOfPoint[p];
					
					if(this.coreDistance[p] > this.candidateDistance[component1])
						continue;
					
					for(j = 0; j < pointIndices2.length; j++) {
						q = pointIndices2[j];
						component2 = this.componentOfPoint[q];
						
						if(this.coreDistance[q] > this.candidateDistance[component1])
							continue;
						
						
						// They belong to different components
						if(component1 != component2) {
							
							d = metric.getPartialDistance(this.data_arr[p], this.data_arr[q]);
							
							mrDist = FastMath.max( 
									// Avoid repeated division overhead
									(alpha == 1.0 ? d : d / alpha), 
									
									// Nested max
									FastMath.max(this.coreDistance[p], 
										this.coreDistance[q]));
							
							if(mrDist < this.candidateDistance[component1]) {
								this.candidateDistance[component1]	= mrDist;
								this.candidateNeighbors[component1]	= q;
								this.candidatePoint[component1]		= p;
							}
						}
					} // end for j
					
					newUpperBound = FastMath.max(newUpperBound, this.candidateDistance[component1]);
					newLowerBound = FastMath.min(newLowerBound, this.candidateDistance[component1]);
				} // end for i
				
				// Calc new bound
				newBound = FastMath.min(newUpperBound, newLowerBound + 2 * node1Info.radius());
				
				// Reassign new bound to min bounds[node1]
				if(newBound < this.bounds[node1]) {
					this.bounds[node1] = newBound;
					
					// propagate bounds up...
					while(node1 > 0) {
						parent = (node1 - 1) / 2;
						left = 2 * parent + 1;
						right = 2 * parent + 2;
						
						newBound = FastMath.max(this.bounds[left], this.bounds[right]);
						if(newBound < this.bounds[parent]) {
							this.bounds[parent] = newBound;
							node1 = parent;
						} else break;
					} // end while
				} // end if inner
			} // end case 1 if
			
			
			// If node is a leaf or smaller than ref node
			else if(node1Info.isLeaf() 
					|| (!node2Info.isLeaf() 
							&& node2Info.radius() > node1Info.radius())) {
				
				left = 2 * node2 + 1;
				right = 2 * node2 + 2;
				
				node2Info = this.node_data[left];
				leftDist = kdTreeMinRDistDual(metric, 
						node1, left, node_bounds, this.numFeatures);
				
				node2Info = this.node_data[right];
				rightDist= kdTreeMinRDistDual(metric,
						node1, right,node_bounds, this.numFeatures);
				
				if(leftDist < rightDist) {
					this.dualTreeTraversal(node1, left);
					this.dualTreeTraversal(node1, right);
					
				} else { // Navigate in opposite order
					this.dualTreeTraversal(node1, right);
					this.dualTreeTraversal(node1, left);
				}
			} // end case 2 if
			
			
			// Node is leaf or smaller than query node
			else {
				left = 2 * node1 + 1;
				right = 2 * node1 + 2;
				
				node1Info = this.node_data[left];
				leftDist = kdTreeMinRDistDual(metric, 
						left, node2, node_bounds, this.numFeatures);
				
				node1Info = this.node_data[right];
				rightDist= kdTreeMinRDistDual(metric,
						right,node2, node_bounds, this.numFeatures);
				
				if(leftDist < rightDist) {
					this.dualTreeTraversal(left, node2);
					this.dualTreeTraversal(right, node2);
					
				} else {
					this.dualTreeTraversal(right, node2);
					this.dualTreeTraversal(left, node2);
				}
			}
			
			
			return 0;
		}
	}
	
	class BallTreeBoruvAlg extends BoruvAlg {
		final double[][] centroidDistances;
		
		BallTreeBoruvAlg() {
			super(false, new BallTree(
				new Array2DRowRealMatrix(outer_tree.getDataRef(), false), 
				leafSize, metric, logger));
			
			// Compute pairwise dist matrix for node_bounds
			centroidDistances = ClustUtils
				.distanceFullMatrix(node_bounds[0], metric);
		}

		@Override
		void computeBounds() {
			int n, i, m;
			
			Neighborhood queryResult =
				TREE.query(data_arr, minSamples, true, true);
		
			double[][] knnDist = queryResult.getDistances();
			int[][] knnIndices = queryResult.getIndices();
			
			// Assign the core distance array...
			this.coreDistance = new double[knnDist.length];
			for(i = 0; i < coreDistance.length; i++)
				coreDistance[i] = knnDist[i][minSamples - 1];
			
			for(n = 0; n < numPoints; n++) {
				for(i = minSamples - 1; i < 0; i++) {
					m = knnIndices[n][i];
					
					if(this.coreDistance[m] <= this.coreDistance[n]) {
						this.candidatePoint[n] 	   = n;
						this.candidateNeighbors[n] = m;
						this.candidateDistance[n]  = this.coreDistance[n];
					}
				}
			}
			
			updateComponents();
			
			for(n = 0; n < numNodes; n++)
				this.bounds[n] = Double.MAX_VALUE;
		}

		@Override
		int dualTreeTraversal(int node1, int node2) {
			int[] pointIndices1, pointIndices2;
			int i, j, p, q, parent //,child1, child2
			;
			
			double nodeDist, d, mrDist, newBound,
				newUpperBound, newLowerBound,
				boundMax, boundMin,
				leftDist, rightDist;
			
			NodeData node1Info = node_data[node1],
					 node2Info = node_data[node2]
					 ,parentInfo, leftInfo, rightInfo
					;
			
			int component1, component2, left, right;
			
			// Distance btwn query and ref nodes
			nodeDist = ballTreeMinDistDual(node1Info.radius(),
					node2Info.radius(), node1, node2, 
					this.centroidDistances);
			
			// If dist < current bound and nodes are not in the
			// same component, we continue
			if(nodeDist < this.bounds[node1]) {
				if(this.componentOfNode[node1] == this.componentOfNode[node2]
					&& this.componentOfNode[node1] >= 0)
					return 0;
				else {
					/*
					 * Pass. This is the only condition in which
					 * the method will continue without exiting early
					 */
				}
			} else
				return 0;
			
			
			
			// If both nodes are leaves
			if(node1Info.isLeaf() && node2Info.isLeaf()) {
				newUpperBound = Double.NEGATIVE_INFINITY;
				newLowerBound = Double.MAX_VALUE;
				newBound = 0.0;
				
				// Build the indices
				pointIndices1 = new int[node1Info.end() - node1Info.start()];
				pointIndices2 = new int[node2Info.end() - node2Info.start()];
				
				// Populate the indices
				for(i = node1Info.start(), j = 0; i < node1Info.end(); i++, j++)
					pointIndices1[j] = this.idx_array[i];
				for(i = node2Info.start(), j = 0; i < node2Info.end(); i++, j++)
					pointIndices2[j] = this.idx_array[i];
				
				
				for(i = 0; i < pointIndices1.length; i++) {
					p = pointIndices1[i];
					component1 = this.componentOfPoint[p];
					
					if(this.coreDistance[p] > this.candidateDistance[component1])
						continue;
					
					for(j = 0; j < pointIndices2.length; j++) {
						q = pointIndices2[j];
						component2 = this.componentOfPoint[q];
						
						if(this.coreDistance[q] > this.candidateDistance[component1])
							continue;
						
						// They belong to different components
						if(component1 != component2) {
							d = metric.getDistance(this.data_arr[p], this.data_arr[q]);
							
							mrDist = FastMath.max( 
									// Avoid repeated division overhead
									(alpha == 1.0 ? d : d / alpha), 
									
									// Nested max
									FastMath.max(this.coreDistance[p], 
										this.coreDistance[q]));
							
							if(mrDist < this.candidateDistance[component1]) {
								this.candidateDistance[component1]	= mrDist;
								this.candidateNeighbors[component1]	= q;
								this.candidatePoint[component1]		= p;
							}
						}
					} // end for j
					
					newUpperBound = FastMath.max(newUpperBound, this.candidateDistance[component1]);
					newLowerBound = FastMath.min(newLowerBound, this.candidateDistance[component1]);
				} // end for i
				
				// Calc new bound
				newBound = FastMath.min(newUpperBound, newLowerBound + 2 * node1Info.radius());
				
				// Reassign new bound to min bounds[node1]
				if(newBound < this.bounds[node1]) {
					this.bounds[node1] = newBound;
					
					// propagate bounds up...
					while(node1 > 0) {
						parent = (node1 - 1) / 2;
						left = 2 * parent + 1;
						right = 2 * parent + 2;
						
						parentInfo = this.node_data[parent];
						leftInfo = this.node_data[left];
						rightInfo = this.node_data[right];
						
						boundMax = FastMath.max(this.bounds[left], this.bounds[right]);
						boundMin = FastMath.min(this.bounds[left] + 2 * (parentInfo.radius() - leftInfo.radius()), 
												this.bounds[right]+ 2 * (parentInfo.radius() -rightInfo.radius()));
						
						if(boundMin > 0)
							newBound = FastMath.min(boundMax, boundMin);
						else
							newBound = boundMax;
						
						if(newBound < this.bounds[parent]) {
							this.bounds[parent] = newBound;
							node1 = parent;
						} else break;
					} // end while
				} // end if inner
			} // end case 1 if
			
			
			// If node is a leaf or smaller than ref node
			else if(node1Info.isLeaf() 
					|| (!node2Info.isLeaf() 
							&& node2Info.radius() > node1Info.radius())) {
				left = 2 * node2 + 1;
				right = 2 * node2 + 2;
				
				node2Info = this.node_data[left];
				leftDist = ballTreeMinDistDual(node1Info.radius(),
						node2Info.radius(), node1, left, this.centroidDistances);
				
				node2Info = this.node_data[right];
				rightDist= ballTreeMinDistDual(node1Info.radius(),
						node2Info.radius(), node1, right, this.centroidDistances);
				
				if(leftDist < rightDist) {
					this.dualTreeTraversal(node1, left);
					this.dualTreeTraversal(node1, right);
					
				} else { // Navigate in opposite order
					this.dualTreeTraversal(node1, right);
					this.dualTreeTraversal(node1, left);
				}
			} // end case 2 if
			
			
			// Node is leaf or smaller than query node
			else {
				left = 2 * node1 + 1;
				right = 2 * node1 + 2;
				
				node1Info = this.node_data[left];
				leftDist = ballTreeMinDistDual(node1Info.radius(),
						node2Info.radius(), left, node2, this.centroidDistances);
				
				node1Info = this.node_data[right];
				rightDist= ballTreeMinDistDual(node1Info.radius(),
						node2Info.radius(), right, node2, this.centroidDistances);
				
				if(leftDist < rightDist) {
					this.dualTreeTraversal(left, node2);
					this.dualTreeTraversal(right, node2);
					
				} else {
					this.dualTreeTraversal(right, node2);
					this.dualTreeTraversal(left, node2);
				}
			}
			
			
			return 0;
		}
	}
	
	final double[][] spanningTree() {
		return alg.spanningTree();
	}
}
