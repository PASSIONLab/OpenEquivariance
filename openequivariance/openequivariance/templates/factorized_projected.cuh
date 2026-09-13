// Receiver-owned forward kernels and edge-owned reverse kernels generated from
// sparse coupling paths.
using int64_t = signed long long;
using int32_t = int;
using scalar_t = {{ scalar }};
using acc_t = {{ scalar }};

constexpr int INPUT_DIM = {{ schedule.input_dim }};
constexpr int EDGE_DIM = {{ schedule.edge_dim }};
constexpr int OUTPUT_DIM = {{ schedule.output_dim }};
constexpr int WEIGHT_DIM = {{ schedule.weight_numel }};
constexpr int CHANNEL_DIM = {{ schedule.channels }};
constexpr int LOGICAL_GROUP_SIZE =
    {{ schedule.launch_config.logical_cohort_width }};
constexpr unsigned int FULL_MASK = 0xffffffffu;

struct EdgeRange {
    int64_t begin;
    int64_t end;
};

__device__ __forceinline__ EdgeRange safe_edge_range(
    const int32_t* row_ptr,
    const int64_t node,
    const int64_t edge_count) {
    const int64_t begin = row_ptr[node] < 0 ? 0 : row_ptr[node];
    int64_t end = row_ptr[node + 1] < begin ? begin : row_ptr[node + 1];
    end = end > edge_count ? edge_count : end;
    return {begin, end};
}

{% macro value_or_zero(active, value) -%}
    {%- if active -%}
        {{ value }}
    {%- else -%}
        scalar_t(0)
    {%- endif -%}
{%- endmacro %}

{# Selectively active tangent reads. Inactive operands render as zero. #}
{% macro value_if_active(input, index, forward) -%}
    {%- if forward -%}
        {%- if forward_jvp_active[input.value] -%}
            {%- if input == Input.X -%}
                tx[{{ index }}]
            {%- elif input == Input.SH -%}
                tsh[{{ index }}]
            {%- elif input == Input.W -%}
                tweights[{{ index }}]
            {%- endif -%}
        {%- else -%}
            scalar_t(0)
        {%- endif -%}
    {%- else -%}
        {%- if backward_jvp_active[input.value] -%}
            {%- if input == Input.X -%}
                tx[{{ index }}]
            {%- elif input == Input.SH -%}
                tsh[{{ index }}]
            {%- elif input == Input.W -%}
                tweights[{{ index }}]
            {%- endif -%}
        {%- else -%}
            scalar_t(0)
        {%- endif -%}
    {%- endif -%}
{%- endmacro %}


{% macro angular_contraction(name, path, output, edge_channel, tangent=false) -%}
    {%- if output.terms | length %}
        {%- for term in output.terms %}
            const int64_t {{ name }}_input_index_{{ loop.index0 }} =
                input_base + {{ path.input_start + term.input_component }}
                + channel * {{ path.input_irrep_dim }};
            const int64_t {{ name }}_edge_index_{{ loop.index0 }} =
                edge_base + {{ path.edge_start + term.edge_component }}
                + {{ edge_channel }} * {{ path.edge_irrep_dim }};
            const scalar_t {{ name }}_input_value_{{ loop.index0 }} =
                x[{{ name }}_input_index_{{ loop.index0 }}];
            const scalar_t {{ name }}_edge_value_{{ loop.index0 }} =
                sh[{{ name }}_edge_index_{{ loop.index0 }}];
            {%- if tangent %}
                const scalar_t {{ name }}_tangent_input_value_{{ loop.index0 }} =
                    {{ value_if_active(
                        Input.X, name ~ "_input_index_" ~ loop.index0, true
                    ) }};
                const scalar_t {{ name }}_tangent_edge_value_{{ loop.index0 }} =
                    {{ value_if_active(
                        Input.SH, name ~ "_edge_index_" ~ loop.index0, true
                    ) }};
            {%- endif %}
        {%- endfor %}

        const scalar_t {{ name }} =
        {%- for term in output.terms %}
            {%- if not loop.first %}
                +
            {%- endif %}

            {%- if tangent %}
                ({{ name }}_tangent_input_value_{{ loop.index0 }}
                    * {{ name }}_edge_value_{{ loop.index0 }}
                    + {{ name }}_input_value_{{ loop.index0 }}
                    * {{ name }}_tangent_edge_value_{{ loop.index0 }})
            {%- else %}
                {{ name }}_input_value_{{ loop.index0 }}
                    * {{ name }}_edge_value_{{ loop.index0 }}
            {%- endif %}
                * {{ cpp_scalar_literal(term.coefficient, scalar) }}
        {%- endfor %}
            ;
    {%- else %}
        const scalar_t {{ name }} = scalar_t(0);
    {%- endif %}
{%- endmacro %}

// One thread owns (receiver node, channel), scans that receiver's CSR edges,
// and directly writes all of its output components.
extern "C" __global__ void oeq_projected_forward(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* row_ptr,
    scalar_t* out) {

    const int64_t node_channel_index =
        int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t node_channel_count = node_count * CHANNEL_DIM;
    if (node_channel_index >= node_channel_count) return;

    const int64_t node = node_channel_index / CHANNEL_DIM;
    const int channel = int(node_channel_index - node * CHANNEL_DIM);
    const int64_t output_base = node * OUTPUT_DIM;

    {%- for slot in schedule.output_slots %}
        scalar_t {{ slot.name }} = scalar_t(0);
    {%- endfor %}

    const EdgeRange edge_range =
        safe_edge_range(row_ptr, node, edge_count);
    for (int64_t e = edge_range.begin; e < edge_range.end; ++e) {
        const int32_t sender = senders[e];
        if (sender < 0 || sender >= node_count) continue;

        const int64_t input_base = int64_t(sender) * INPUT_DIM;
        const int64_t edge_base = e * EDGE_DIM;
        const int64_t weight_base = e * WEIGHT_DIM;

        {% for path in schedule.paths %}
            {% for edge_channel in range(path.edge_mul) %} {
                const int weight_index =
                    {{ path.weight_start + edge_channel * schedule.channels }} + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];

                {% for output in path.outputs %} {
                    {{ angular_contraction(
                        "angular_value", path, output, edge_channel
                    ) }}
                    {{ output.accumulator.name }} +=
                        radial_weight * angular_value;
                } {%- endfor %}
            } {%- endfor %}
        {%- endfor %}
    }
    {% for slot in schedule.output_slots %} {
        const int64_t output_index =
            output_base + {{ slot.array_index }}
            + channel * {{ slot.irrep_dim }};

        out[output_index] = {{ slot.name }};
    } {%- endfor %}
}

// The same (receiver node, channel) ownership computes the forward tangent.
extern "C" __global__ void oeq_projected_forward_jvp(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* row_ptr,
    const scalar_t* tx,
    const scalar_t* tsh,
    const scalar_t* tweights,
    scalar_t* out) {

    const int64_t node_channel_index =
        int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t node_channel_count = node_count * CHANNEL_DIM;
    if (node_channel_index >= node_channel_count) return;

    const int64_t node = node_channel_index / CHANNEL_DIM;
    const int channel = int(node_channel_index - node * CHANNEL_DIM);
    const int64_t output_base = node * OUTPUT_DIM;

    {%- for slot in schedule.output_slots %}
        scalar_t tangent_{{ slot.name }} = scalar_t(0);
    {%- endfor %}

    const EdgeRange edge_range =
        safe_edge_range(row_ptr, node, edge_count);
    for (int64_t e = edge_range.begin; e < edge_range.end; ++e) {
        const int32_t sender = senders[e];
        if (sender < 0 || sender >= node_count) continue;

        const int64_t input_base = int64_t(sender) * INPUT_DIM;
        const int64_t edge_base = e * EDGE_DIM;
        const int64_t weight_base = e * WEIGHT_DIM;
        {% for path in schedule.paths %}
            {% for edge_channel in range(path.edge_mul) %} {
                const int weight_index =
                    {{ path.weight_start + edge_channel * schedule.channels }} + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    {{ value_if_active(Input.W, "radial_weight_index", true) }};

                {% for output in path.outputs %} {
                    {{ angular_contraction(
                        "angular_value", path, output, edge_channel
                    ) }}
                    {{ angular_contraction(
                        "tangent_angular_value", path, output, edge_channel,
                        tangent=true
                    ) }}
                    tangent_{{ output.accumulator.name }} +=
                        tangent_weight * angular_value
                        + radial_weight * tangent_angular_value;
                } {%- endfor %}
            } {%- endfor %}
        {%- endfor %}
    }
    {% for slot in schedule.output_slots %} {
        const int64_t output_index =
            output_base + {{ slot.array_index }}
            + channel * {{ slot.irrep_dim }};

        out[output_index] = tangent_{{ slot.name }};
    } {%- endfor %}
}

// One logical thread group owns an edge, including on HIP wavefront-64.
// Each lane owns channels separated by LOGICAL_GROUP_SIZE.
extern "C" __global__ void oeq_projected_backward(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    scalar_t* dx,
    scalar_t* dsh,
    scalar_t* dweights) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count)
        return;

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
        {%- for slot in schedule.input_gradient_slots %}
            acc_t {{ slot.name }} = acc_t(0);
        {%- endfor %}

        // Path weight intervals are disjoint. This lane directly owns each
        // dweights[e, weight] element generated below.
        {% for path in schedule.paths %}
            {% for edge_channel in range(path.edge_mul) %} {
                const int weight_index =
                    {{ path.weight_start + edge_channel * schedule.channels }} + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                {% for output in path.outputs %} {
                    const int64_t output_index =
                        output_base + {{ output.accumulator.array_index }}
                        + channel * {{ output.accumulator.irrep_dim }};
                    const scalar_t output_gradient = dout[output_index];

                    // Accumulate weight, sender-feature, and edge-feature adjoints.
                    {% for term in output.terms %} {
                        const int64_t input_index =
                            input_base + {{ path.input_start + term.input_component }}
                            + channel * {{ path.input_irrep_dim }};
                        const int edge_component_index =
                            {{ path.edge_start + term.edge_component }}
                            + {{ edge_channel }} * {{ path.edge_irrep_dim }};

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        {{ term.input_accumulator.name }} += output_gradient
                            * radial_weight * edge_value * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                    } {%- endfor %}
                } {%- endfor %}
                dweights[radial_weight_index] = scalar_t(weight_gradient);
            } {%- endfor %}
        {%- endfor %}

        // Paths are combined locally for this (edge, channel, input component),
        // but different receiver-owned edges can contribute to the same sender.
        {%- for slot in schedule.input_gradient_slots %}
            {{ atomic_add }}(
                dx + input_base + {{ slot.array_index }}
                    + channel * {{ slot.irrep_dim }},
                scalar_t({{ slot.name }}));
        {%- endfor %}
    }

    // Reduce within the logical 32-thread edge owner. Lane 0 writes dsh[e, :].
    for (int offset = 16; offset > 0; offset >>= 1)
        for (int component = 0; component < EDGE_DIM; ++component)
            edge_gradient[component] +=
                {{ shfl_down_32("edge_gradient[component]", "offset") }};

    if (lane == 0)
        for (int component = 0; component < EDGE_DIM; ++component)
            dsh[edge_base + component] = scalar_t(edge_gradient[component]);
}

// Mixed derivative of the edge-owned reverse pass, with the same logical
// 32-thread group and channel ownership as oeq_projected_backward.
extern "C" __global__ void oeq_projected_backward_jvp(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    const scalar_t* tx,
    const scalar_t* tsh,
    const scalar_t* tweights,
    const scalar_t* tdout,
    scalar_t* tdx,
    scalar_t* tdsh,
    scalar_t* tdweights) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count)
        return;

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    acc_t tangent_edge_gradient[EDGE_DIM] = {acc_t(0)};
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
        {%- for slot in schedule.input_gradient_slots %}
            acc_t {{ slot.name }} = acc_t(0);
        {%- endfor %}

        // Path weight intervals are disjoint. This lane directly owns each
        // tdweights[e, weight] element generated below.
        {% for path in schedule.paths %}
            {% for edge_channel in range(path.edge_mul) %} {
                const int weight_index =
                    {{ path.weight_start + edge_channel * schedule.channels }} + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                const scalar_t tangent_weight =
                    {{ value_if_active(Input.W, "radial_weight_index", false) }};

                acc_t tangent_weight_gradient = acc_t(0);
                {% for output in path.outputs %} {
                    const int64_t output_index =
                        output_base + {{ output.accumulator.array_index }}
                        + channel * {{ output.accumulator.irrep_dim }};
                    const scalar_t output_gradient = dout[output_index];
                    const scalar_t tangent_output_gradient =
                        {{ value_or_zero(
                            backward_jvp_dout_active, "tdout[output_index]") }};

                    {% for term in output.terms %} {
                        const int64_t input_index =
                            input_base + {{ path.input_start + term.input_component }}
                            + channel * {{ path.input_irrep_dim }};
                        const int edge_component_index =
                            {{ path.edge_start + term.edge_component }}
                            + {{ edge_channel }} * {{ path.edge_irrep_dim }};
                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];
                        const scalar_t tangent_input_value =
                            {{ value_if_active(Input.X, "input_index", false) }};
                        const scalar_t tangent_edge_value =
                            {{ value_if_active(Input.SH, "edge_index", false) }};

                        {{ term.input_accumulator.name }} +=
                            (tangent_output_gradient * radial_weight * edge_value
                             + output_gradient * tangent_weight * edge_value
                             + output_gradient * radial_weight * tangent_edge_value)
                            * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        tangent_edge_gradient[edge_component_index] +=
                            (tangent_output_gradient * radial_weight * input_value
                             + output_gradient * tangent_weight * input_value
                             + output_gradient * radial_weight * tangent_input_value)
                            * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        tangent_weight_gradient +=
                            (tangent_output_gradient * input_value * edge_value
                             + output_gradient * tangent_input_value * edge_value
                             + output_gradient * input_value * tangent_edge_value)
                            * {{ cpp_scalar_literal(term.coefficient, scalar) }};

                    } {%- endfor %}
                } {%- endfor %}

                tdweights[radial_weight_index] =
                    scalar_t(tangent_weight_gradient);

            } {%- endfor %}
        {%- endfor %}

        // Paths are combined locally for this (edge, channel, input component),
        // but different receiver-owned edges can contribute to the same sender.
        {%- for slot in schedule.input_gradient_slots %}
            {{ atomic_add }}(
                tdx + input_base + {{ slot.array_index }}
                    + channel * {{ slot.irrep_dim }},
                scalar_t({{ slot.name }}));
        {%- endfor %}
    }

    // Reduce within the logical 32-thread edge owner. Lane 0 writes tdsh[e, :].
    for (int offset = 16; offset > 0; offset >>= 1)
        for (int component = 0; component < EDGE_DIM; ++component)
            tangent_edge_gradient[component] +=
                {{ shfl_down_32("tangent_edge_gradient[component]", "offset") }};

    if (lane == 0)
        for (int component = 0; component < EDGE_DIM; ++component)
            tdsh[edge_base + component] =
                scalar_t(tangent_edge_gradient[component]);
}
