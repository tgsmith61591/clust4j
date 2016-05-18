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

package com.clust4j.optimize;

/**
 * A wrapper class to optimize some function. Usually
 * instantiated via anonymous classing
 * @author Taylor G Smith
 * @param <T>
 */
public interface OptimizableCaller {
	/**
	 * The function whose output to minimize
	 * @param val
	 * @return
	 */
	public double doCall(double val);
}
