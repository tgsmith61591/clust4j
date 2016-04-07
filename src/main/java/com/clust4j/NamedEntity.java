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
package com.clust4j;

/**
 * 
 * Models or any {@link com.clust4j.log.Loggable}
 * that should be able to "say their name" should 
 * implement this method.
 * 
 * <p>Other considered names:
 * <p><ul><tt>SelfProfessant</tt></ul>
 * <p><ul><tt>Parrot</tt></ul>
 * <p><ul><tt>EchoChamber</tt></ul>
 * 
 * :-)
 * 
 * <p>
 * @author Taylor G Smith
 */
public interface NamedEntity {
	public String getName();
}
