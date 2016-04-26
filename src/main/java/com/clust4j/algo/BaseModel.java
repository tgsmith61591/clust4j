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

import java.text.NumberFormat;

import com.clust4j.Clust4j;
import com.clust4j.utils.SynchronicityLock;
import com.clust4j.utils.TableFormatter;

abstract public class BaseModel extends Clust4j implements java.io.Serializable {
	private static final long serialVersionUID = 4707757741169405063L;
	public final static TableFormatter formatter;
	
	// Initializers
	static {
		NumberFormat nf = NumberFormat.getInstance(TableFormatter.DEFAULT_LOCALE);
		nf.setMaximumFractionDigits(5);
		formatter = new TableFormatter(nf);
		formatter.leadWithEmpty = false;
		formatter.setWhiteSpace(1);
	}
	
	/** The lock to synchronize on for fits */
	protected final Object fitLock = new SynchronicityLock();

	/** This should be synchronized and thread-safe */
	protected abstract BaseModel fit();
}
