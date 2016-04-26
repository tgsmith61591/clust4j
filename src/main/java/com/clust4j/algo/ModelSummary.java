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

/**
 * The {@link com.clust4j.utils.TableFormatter} uses this class
 * for pretty printing of various models' fit summaries.
 * @author Taylor G Smith
 */
public class ModelSummary extends ArrayList<Object[]> {
	private static final long serialVersionUID = -8584383967988199855L;
	
	public ModelSummary(final Object[] ... objs) {
		super();
		for(Object[] o: objs)
			this.add(o);
	}
}
