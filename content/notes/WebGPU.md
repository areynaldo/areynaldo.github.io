---
title: "WebGPU"
date: 2026-01-27
summary: "TODO"
description: "TODO"
math: true
---

#### Hello Triangle

<canvas></canvas>

<script type=module>
async function main() {
    const adapter = await navigator?.gpu.requestAdapter();

    const device = await adapter?.requestDevice();
    if (!device) {
        fail('need a browser that supports WebGPU');
        return;
    }

    const canvas = document.querySelector('canvas');
    const context = canvas.getContext('webgpu');
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentationFormat,
    });

    const module = device.createShaderModule({
    label: 'our hardcoded red triangle shaders',
    code: /* wgsl */ `
      @vertex fn vs(
        @builtin(vertex_index) vertexIndex : u32
      ) -> @builtin(position) vec4f {
        let pos = array(
          vec2f( 0.0,  0.5),  // top center
          vec2f(-0.5, -0.5),  // bottom left
          vec2f( 0.5, -0.5)   // bottom right
        );

        return vec4f(pos[vertexIndex], 0.0, 1.0);
      }

      @fragment fn fs() -> @location(0) vec4f {
        return vec4f(0.0, 0.0, 0.0, 1.0);
      }
    `,
  });

    const pipeline = device.createRenderPipeline({
        label: 'our hardcoded red triangle pipeline',
        layout: 'auto',
        vertex: {
            module,
        },
        fragment: {
            module,
            targets: [{ format: presentationFormat }],
        },
    });


  const renderPassDescriptor = {
    label: 'our basic canvas renderPass',
    colorAttachments: [
      {
        // view: <- to be filled out when we render
        clearValue: [1.0, 1.0, 1.0, 1.0],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

    function render() {
        // Get the current texture from the canvas context and
        // set it as the texture to render to.
        renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture();

        // make a command encoder to start encoding commands
        const encoder = device.createCommandEncoder({ label: 'our encoder' });

        // make a render pass encoder to encode render specific commands
        const pass = encoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(pipeline);
        pass.draw(3);  // call our vertex shader 3 times
        pass.end();

        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
    }

    render();
}
main();
</script>

```js
async function main() {

    // request adapter
    const adapter = await navigator.gpu?.requestAdapter();

    // request device
    const device = await adapter?.requestDevice();
    if (!device) {
        fail("need a browser that supports WebGPU");
        return;
    }

    // get canvas
    const canvas = document.querySelector("canvas");

    // get webgpu context from canvas
    const context = canvas.getContext("webgpu");

    // set get recommended """ideal/recommended""" format (should be faster)
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: presentationFormat,
    });

    // create shader
    const module = device.createShaderModule({
        label: "our hardcoded red triangle shaders",
        code: `
        @vertex fn vs(                                // vertex shader function
            @builtin(vertex_index) vertexIndex : u32  // builtin vertex iterator
        ) -> @builtin(position) vec4f                 // returns vertex positon [-1.0..1.0]
        {
            let pos = array(
            vec2f( 0.0,  0.5),  // top center
            vec2f(-0.5, -0.5),  // bottom left
            vec2f( 0.5, -0.5)   // bottom right
            );

            return vec4f(pos[vertexIndex], 0.0, 1.0); // position
        }

        @fragment fn fs() -> @location(0) vec4f {   // fragment shader function
                                                    // returns vec4 at location 0
                                                    // this will write to the first render target
            return vec4f(0.0, 0.0, 0.0, 1.0);       // color
        }
        `,
    });

    const pipeline = device.createRenderPipeline({  // define pipeline
        label: "our hardcoded red triangle pipeline",
        layout: "auto",                             // data layout (not data in this example)
        vertex: {
            module,                                 // set vertex shader
        },
        fragment: {
            module,                                 // set fragmaent shader
            targets: [{
                format: presentationFormat          // use presentation format
            }],
        },
    });

    const renderPassDescriptor = {                  // define render pass
        label: "our basic canvas renderPass",
        colorAttachments: [                         // color step
        {
            clearValue: [1.0, 1.0, 1.0, 1.0],       // background color
            loadOp: "clear",                        // load operation
            storeOp: "store",                       // store operation
        },
        ],
    };

    function render() {
        // Get the current texture from the canvas context and
        // set it as the texture to render to.
        renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture();

        // make a command encoder to start encoding commands
        const encoder = device.createCommandEncoder({
            label: "our encoder"
        });

        // make a render pass encoder to encode render specific commands
        const pass = encoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(pipeline);
        pass.draw(3); // call our vertex shader 3 times
        pass.end();

        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
    }

    render();
}
```

#### Compute Shader
```js
async function main() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) {
        fail("need a browser that supports WebGPU");
        return;
    }

    const module = device.createShaderModule({
        lable: 'doubling compute module',
        code: `
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(1) fn computeSomething(
                @builtin(global_invocation_id) id: vec3u
            ) {
                let i = id.x;
                data[i] = data[i] * 2;
            }
        `
    })

    const pipeline = device.createComputePipeline({
        label: 'doubling compute pipeline',
        layout: 'auto',
        compute: {
            module,
        }
    })

    const input = new Float32Array([1,2,3]); // host araray

    // create a buffer on the GPU to hold our computation
    // input and output
    const workBuffer = device.createBuffer({
        label: 'work buffer',
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // Copy our input data to that buffer
    device.queue.writeBuffer(workBuffer, 0, input);

    // create a buffer on the GPU to hold our computation
    // input and output
    const resultBuffer = device.createBuffer({
        label: 'work buffer',
        size: input.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // setup bind group
    // tell the shader which buffer to use
    const bindGroup = device.createBindGroup({
        label: 'bindGroup for work buffer',
        layout: pipeline.getBindGroupLayout(0), // '@group(0)' in the shader
        entries: [{
            binding: 0,                         // '@group(0) @binding(0)' in the shader
            resource: workBuffer
        },],
    });

    // encode commands
    const encoder = device.createCommandEncoder({
        label = 'doubling encoder',
    })
    const pass = encoder.beginComputePass({
        label = 'doubling compute pass',
    })
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);            // '@group(0)' in the shader
    pass.dispatchWorkgroups(input.length);
    pass.end();

    // copy results to mappable buffer
    encoder.copyToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size);

    // finish encoding and submit the commands
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    // read the results
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(resultBuffer.getMappedRange());

    console.log('input', input);
    console.log('result', result);

    resultBuffer.unmap();
}
```
