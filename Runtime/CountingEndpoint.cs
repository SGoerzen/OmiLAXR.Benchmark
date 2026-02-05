/*
* SPDX-License-Identifier: AGPL-3.0-or-later
* Copyright (C) 2025 Sergej Görzen <sergej.goerzen@gmail.com>
* This file is part of OmiLAXR.
*/
using OmiLAXR.Composers;
using OmiLAXR.Endpoints;

namespace OmiLAXR.Benchmark
{
    /// <summary>
    /// Minimal endpoint used for benchmarking pipelines without network or storage overhead.
    /// Always reports successful transfer for each statement it receives.
    /// </summary>
    public class CountingEndpoint : Endpoint
    {
        /// <summary>
        /// Handles a statement send by immediately returning success without side effects.
        /// </summary>
        /// <param name="statement">Statement to be "sent" by this endpoint</param>
        /// <returns>Always <see cref="TransferCode.Success"/></returns>
        protected override TransferCode HandleSending(IStatement statement)
        {
            return TransferCode.Success;
        }
    }

}