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

import com.clust4j.Clust4j;
import com.clust4j.utils.SynchronicityLock;

abstract public class BaseModel extends Clust4j implements java.io.Serializable {
	private static final long serialVersionUID = 4707757741169405063L;
	
	/** The lock to synchronize on for fits */
	final Object fitLock = new SynchronicityLock();

	/** This should be synchronized and thread-safe */
	public abstract BaseModel fit();
}
