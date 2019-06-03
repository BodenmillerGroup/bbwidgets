import ipywidgets as wx
import matplotlib.pyplot as plt
import numpy as np
import math
import traitlets
import traittypes

from functools import partial
from matplotlib.colors import colorConverter, LinearSegmentedColormap


# noinspection PyUnusedLocal
def _validate_image_trait(trait, value):
    if value is None:
        raise traitlets.TraitError()
    if not isinstance(value, np.ndarray):
        raise traitlets.TraitError()
    if value.ndim != 3:
        raise traitlets.TraitError()
    return value


class MultichannelView(wx.GridBox):
    GALLERY_TITLE = 'Gallery'
    CHANNELS_PANEL_TITLE = 'Channels'
    CHANNEL_PROPERTIES_PANEL_TITLE = 'Channel properties'
    IMAGE_NAME_FMT = 'Image {image_id:d}'
    CHANNEL_NAME_FMT = 'Channel {channel_id:d}'
    DEFAULT_CHANNEL_STATE = False
    DEFAULT_CHANNEL_COLOR = 'white'
    GALLERY_FIGSIZE = (4, 4)
    IMAGE_FIGSIZE = (12, 12)

    images = traitlets.List(trait=traittypes.Array().valid(_validate_image_trait))
    image_names = traitlets.List(trait=traitlets.Unicode(), allow_none=True)
    channel_names = traitlets.List(trait=traitlets.Unicode(), allow_none=True)
    channel_vmins = traitlets.List(trait=traitlets.Float())
    channel_vmaxs = traitlets.List(trait=traitlets.Float())
    channel_states = traitlets.List(trait=traitlets.Bool())
    channel_colors = traitlets.List(trait=traitlets.Unicode())
    num_histogram_bins = traitlets.Int(100)
    selected_channel = traitlets.Int(0)
    num_columns = traitlets.Int(4)

    def __init__(self, **kwargs):
        # channels panel
        self._channel_list_box = wx.VBox([], layout=wx.Layout(border='1px solid grey'))
        channels_panel = wx.VBox([
            wx.Label(value=self.CHANNELS_PANEL_TITLE, layout=wx.Layout(flex='0 0 auto')),
            self._channel_list_box
        ], layout=wx.Layout(grid_area='channels'))
        self._channel_selection_dropdown = wx.Dropdown(layout=wx.Layout(width='calc(100% - 4px)', margin='2px'))
        self._channel_selection_dropdown.observe(self._on_channel_selection_dropdown_value_change, 'value')
        self._selected_channel_histogram_output = wx.Output()
        self._selected_channel_range_slider = wx.FloatRangeSlider(continuous_update=False, readout_format='.0f',
                                                                  layout=wx.Layout(width='calc(100% - 4px)',
                                                                                   margin='2px'))
        self._selected_channel_range_slider.observe(self._on_selected_channel_range_slider_value_change, 'value')
        # channel properties panel
        channel_properties_panel = wx.VBox([
            wx.Label(value=self.CHANNEL_PROPERTIES_PANEL_TITLE, layout=wx.Layout(flex='0 0 auto')),
            self._channel_selection_dropdown,
            self._selected_channel_histogram_output,
            self._selected_channel_range_slider
        ], layout=wx.Layout(grid_area='channel_properties'))
        # tab container
        self._gallery_tab_outputs = []
        self._gallery_tab = wx.GridBox(layout=wx.Layout(grid_auto_rows='minmax(min-content, max-content)'))
        self._tab_panel = wx.Tab([self._gallery_tab], layout=wx.Layout(grid_area='main'))
        self._tab_panel.observe(self._on_tab_panel_selected_index_change, 'selected_index')
        self._tab_panel.set_title(0, self.GALLERY_TITLE)
        # Superclass initialization
        kwargs['layout'] = wx.Layout(grid_template_rows='240px auto', grid_template_columns='240px auto',
                                     grid_template_areas='"channels main" "channel_properties main"')
        super(MultichannelView, self).__init__([self._tab_panel, channels_panel, channel_properties_panel], **kwargs)
        # initialize channel properties
        self._channel_mins = None
        self._channel_maxs = None
        self._channel_hists = None
        self._channel_bin_edges = None
        self._update_channel_properties()
        self._update_channel_histograms()
        # prepare tabs
        self._image_tab_indices = {}
        self._image_tab_outputs = {}
        self._image_tab_show_tiles = {}
        self._image_tab_requires_refresh = {}
        self._update_gallery_tab_outputs()
        # refresh user interface
        self._refresh_channel_list_box()
        self._refresh_channel_selection_dropdown_options()
        self._refresh_channel_selection_dropdown_value()
        self._refresh_selected_channel_histogram()
        self._refresh_gallery_tab_num_columns()
        self._refresh_gallery_tab()

    def get_num_images(self):
        if self.images is not None:
            return len(self.images)
        return 0

    def get_num_channels(self):
        if self.images is not None and len(self.images) > 0:
            return self.images[0].shape[0]
        return 0

    def get_image_name(self, image_id):
        if self.image_names is not None and image_id < len(self.image_names):
            return self.image_names[image_id]
        return self.IMAGE_NAME_FMT.format(image_id=image_id)

    def get_image_names(self):
        return [self.get_image_name(image_id) for image_id in range(self.get_num_images())]

    def get_channel_name(self, channel_id):
        if self.channel_names is not None and channel_id < len(self.channel_names):
            return self.channel_names[channel_id]
        return self.CHANNEL_NAME_FMT.format(channel_id=channel_id)

    def get_channel_names(self):
        return [self.get_channel_name(channel_id) for channel_id in range(self.get_num_channels())]

    def get_active_channels(self):
        return [channel_id for channel_id in range(self.get_num_channels()) if self.channel_states[channel_id]]

    def get_additive_image(self, image_id):
        image = self.images[image_id]
        additive_image = np.zeros(shape=(image.shape[1], image.shape[2], 4), dtype=image.dtype)
        for channel_id in range(self.get_num_channels()):
            if self.channel_states[channel_id]:
                channel_image = self.get_channel_image(image_id, channel_id)
                channel_color = colorConverter.to_rgb(self.channel_colors[channel_id])
                channel_colormap = LinearSegmentedColormap.from_list(None, [(0, 0, 0), channel_color])
                additive_image += channel_colormap(channel_image)
        return np.clip(additive_image, 0, 1, out=additive_image)

    def get_channel_image(self, image_id, channel_id):
        channel_image = self.images[image_id][channel_id] - self.channel_vmins[channel_id]
        channel_image /= self.channel_vmaxs[channel_id] - self.channel_vmins[channel_id]
        return np.clip(channel_image, 0, 1, out=channel_image)

    def _update_channel_properties(self):
        num_channels = self.get_num_channels()
        self._channel_mins = [np.amin([img[c].min() for img in self.images]).item() for c in range(num_channels)]
        self._channel_maxs = [np.amax([img[c].max() for img in self.images]).item() for c in range(num_channels)]
        if self.selected_channel >= num_channels:
            self.selected_channel = 0
        self.channel_states = self.channel_states[:num_channels]
        if len(self.channel_states) < num_channels:
            self.channel_states += [self.DEFAULT_CHANNEL_STATE] * (num_channels - len(self.channel_states))
        self.channel_colors = self.channel_colors[:num_channels]
        if len(self.channel_colors) < num_channels:
            self.channel_colors += [self.DEFAULT_CHANNEL_COLOR] * (num_channels - len(self.channel_colors))
        self.channel_vmins = self.channel_vmins[:num_channels]
        for i, (channel_min, channel_vmin) in enumerate(zip(self._channel_mins, self.channel_vmins)):
            if channel_vmin < channel_min:
                self.channel_vmins[i] = channel_min
        if len(self.channel_vmins) < num_channels:
            self.channel_vmins += self._channel_mins[len(self.channel_vmins):]
        self.channel_vmaxs = self.channel_vmaxs[:num_channels]
        for i, (channel_max, channel_vmax) in enumerate(zip(self._channel_maxs, self.channel_vmaxs)):
            if channel_vmax > channel_max:
                self.channel_vmaxs[i] = channel_max
        if len(self.channel_vmaxs) < num_channels:
            self.channel_vmaxs += self._channel_maxs[len(self.channel_vmaxs):]

    def _update_channel_histograms(self):
        self._channel_hists = []
        self._channel_bin_edges = []
        for channel_id, channel_range in enumerate(zip(self._channel_mins, self._channel_maxs)):
            channel_hist = np.zeros(self.num_histogram_bins)
            channel_bin_edges = None
            for image in self.images:
                hist, bin_edges = np.histogram(image[channel_id], bins=self.num_histogram_bins, range=channel_range)
                channel_bin_edges = bin_edges
                channel_hist += hist
            self._channel_hists.append(channel_hist)
            self._channel_bin_edges.append(channel_bin_edges)

    def _update_gallery_tab_outputs(self):
        num_images = self.get_num_images()
        for output in self._gallery_tab_outputs[num_images:]:
            output.clear_output()
            output.close()
        self._gallery_tab_outputs = self._gallery_tab_outputs[:num_images]
        for i in range(len(self._gallery_tab_outputs), num_images):
            self._gallery_tab_outputs.append(wx.Output())

    def _create_image_tab(self, image_id):
        output = wx.Output()
        self._image_tab_outputs[image_id] = output
        self._image_tab_show_tiles[image_id] = False
        self._image_tab_requires_refresh[image_id] = True
        save_plot_button = wx.Button(description='Save plot')
        save_plot_button.on_click(partial(self._on_image_tab_save_plot_button_click, image_id=image_id))
        toggle_tiles_button = wx.Button(description='Toggle tiles')
        toggle_tiles_button.on_click(partial(self._on_image_tab_toggle_tiles_button_click, image_id=image_id))
        close_button = wx.Button(description='Close')
        close_button.on_click(partial(self._on_image_tab_close_button_click, image_id=image_id))
        left_toolbar = wx.HBox([save_plot_button])
        right_toolbar = wx.HBox([toggle_tiles_button, close_button])
        toolbar = wx.HBox([left_toolbar, right_toolbar], layout=wx.Layout(justify_content='space-between'))
        return wx.VBox([output, toolbar])

    def _close_image_tab(self, image_id):
        image_tab_index = self._image_tab_indices[image_id]
        if self._tab_panel.selected_index >= image_tab_index:
            self._tab_panel.selected_index -= 1
        self._tab_panel.children = [c for i, c in enumerate(self._tab_panel.children) if i != image_tab_index]
        output = self._image_tab_outputs[image_id]
        output.clear_output()
        output.close()
        del self._image_tab_outputs[image_id]
        del self._image_tab_show_tiles[image_id]
        del self._image_tab_requires_refresh[image_id]
        del self._image_tab_indices[image_id]

    def _close_image_tabs(self):
        image_ids = list(self._image_tab_indices.keys())
        for image_id in image_ids:
            self._close_image_tab(image_id)

    def _refresh_channel_list_box(self):
        channel_boxes = []
        for channel_id in range(self.get_num_channels()):
            checkbox = wx.Checkbox(value=self.channel_states[channel_id], description=self.get_channel_name(channel_id),
                                   indent=False, layout=wx.Layout(width='calc(100% - 78px)', margin='2px'))
            checkbox.observe(partial(self._on_channel_checkbox_value_change, channel_id=channel_id), 'value')
            colorpicker = wx.ColorPicker(value=self.channel_colors[channel_id], concise=True,
                                         layout=wx.Layout(width='28px', margin='2px'))
            colorpicker.observe(partial(self._on_channel_colorpicker_value_change, channel_id=channel_id), 'value')
            properties_button = wx.Button(icon='fa-cog', layout=wx.Layout(width='28px', margin='2px'))
            properties_button.on_click(partial(self._on_channel_properties_button_click, channel_id=channel_id))
            channel_box = wx.HBox([checkbox, colorpicker, properties_button], layout=wx.Layout(min_height='30px'))
            channel_boxes.append(channel_box)
        self._channel_list_box.children = channel_boxes

    def _refresh_channel_selection_dropdown_options(self):
        self._channel_selection_dropdown.options = self.get_channel_names()

    def _refresh_channel_selection_dropdown_value(self):
        if self.selected_channel < len(self._channel_selection_dropdown.options):
            self._channel_selection_dropdown.value = self.get_channel_name(self.selected_channel)

    def _refresh_selected_channel_histogram(self):
        self._selected_channel_histogram_output.clear_output(wait=True)
        if self.selected_channel < self.get_num_channels():
            hist = self._channel_hists[self.selected_channel]
            bin_edges = self._channel_bin_edges[self.selected_channel]
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = 0.8 * (bin_edges[1] - bin_edges[0])
            with self._selected_channel_histogram_output:
                fig, ax = plt.subplots()
                ax.bar(centers, hist, width=width, align='center')
                plt.show()

    def _refresh_selected_channel_value_range_slider(self):
        if self.selected_channel < self.get_num_channels():
            self._selected_channel_range_slider.min = self._channel_mins[self.selected_channel]
            self._selected_channel_range_slider.max = self._channel_maxs[self.selected_channel]
            self._selected_channel_range_slider.value = (
                self.channel_vmins[self.selected_channel], self.channel_vmaxs[self.selected_channel])

    def _refresh_gallery_tab_num_columns(self):
        self._gallery_tab.layout.grid_template_columns = 'repeat({:d}, 1fr)'.format(self.num_columns)

    def _refresh_gallery_tab(self):
        image_boxes = []
        for image_id in range(self.get_num_images()):
            image = self.get_additive_image(image_id)
            button = wx.Button(description=self.get_image_name(image_id), layout=wx.Layout(width='calc(100% - 4px)'))
            button.on_click(partial(self._on_image_button_click, image_id=image_id))
            output = self._gallery_tab_outputs[image_id]
            output.clear_output(wait=True)
            with output:
                fig, ax = plt.subplots(figsize=self.GALLERY_FIGSIZE)
                ax.set_axis_off()
                ax.imshow(image)
                plt.show()
            image_box = wx.VBox([button, output], layout=wx.Layout(align_items='center'))
            image_boxes.append(image_box)
        self._gallery_tab.children = image_boxes

    def _refresh_image_tab_titles(self):
        for image_id, tab_index in self._image_tab_indices.items():
            self._tab_panel.set_title(tab_index, self.get_image_name(image_id))

    def _refresh_image_tab(self, image_id):
        output = self._image_tab_outputs[image_id]
        output.clear_output(wait=True)
        with output:
            if self._image_tab_show_tiles[image_id]:
                active_channels = self.get_active_channels()
                nrows = int(math.ceil(len(active_channels) / self.num_columns))
                fix, axes = plt.subplots(nrows=nrows, ncols=self.num_columns, squeeze=False, figsize=self.IMAGE_FIGSIZE)
                for ax in axes.flatten():
                    ax.set_axis_off()
                for channel_id, channel_ax in zip(active_channels, axes.flatten()):
                    channel_image = self.get_channel_image(image_id, channel_id)
                    channel_ax.set_title(self.get_channel_name(channel_id))
                    channel_ax.imshow(channel_image, cmap='binary_r')
            else:
                image = self.get_additive_image(image_id)
                fig, ax = plt.subplots(figsize=self.IMAGE_FIGSIZE)
                ax.set_axis_off()
                ax.imshow(image)
            plt.show()
        self._image_tab_requires_refresh[image_id] = False

    def _refresh_image_tabs(self):
        for image_id, image_tab_index in self._image_tab_indices.items():
            self._image_tab_requires_refresh[image_id] = True
            if image_tab_index == self._tab_panel.selected_index:
                self._refresh_image_tab(image_id)

    def _on_channel_selection_dropdown_value_change(self, change):
        self.selected_channel = self.get_channel_names().index(change.new)

    def _on_channel_checkbox_value_change(self, change, channel_id):
        self.selected_channel = channel_id
        self.channel_states[channel_id] = change.new
        # list item change --> observer does not fire
        if change.new != change.old:
            self._observe_channel_states(None)

    def _on_channel_colorpicker_value_change(self, change, channel_id):
        self.channel_colors[channel_id] = change.new
        # list item change --> observer does not fire
        if change.new != change.old:
            self._observe_channel_colors(None)

    # noinspection PyUnusedLocal
    def _on_channel_properties_button_click(self, button, channel_id):
        self.selected_channel = channel_id

    def _on_selected_channel_range_slider_value_change(self, change):
        self.channel_vmins[self.selected_channel] = change.new[0]
        self.channel_vmaxs[self.selected_channel] = change.new[1]
        # list item change --> observer does not fire
        if change.new != change.old:
            self._observe_channel_range(None)

    # noinspection PyUnusedLocal
    def _on_image_button_click(self, button, image_id):
        if image_id in self._image_tab_indices:
            self._tab_panel.selected_index = self._image_tab_indices[image_id]
        else:
            image_tab = self._create_image_tab(image_id)
            image_tab_index = len(self._tab_panel.children)
            self._tab_panel.children = self._tab_panel.children + (image_tab,)
            self._tab_panel.set_title(image_tab_index, self.get_image_name(image_id))
            self._tab_panel.selected_index = image_tab_index
            self._image_tab_indices[image_id] = image_tab_index
            self._refresh_image_tab(image_id)

    def _on_tab_panel_selected_index_change(self, change):
        for image_id, image_tab_index in self._image_tab_indices.items():
            if image_tab_index == change.new and self._image_tab_requires_refresh[image_id]:
                self._refresh_image_tab(image_id)

    # noinspection PyUnusedLocal
    def _on_image_tab_save_plot_button_click(self, button, image_id):
        pass  # TODO

    # noinspection PyUnusedLocal
    def _on_image_tab_toggle_tiles_button_click(self, button, image_id):
        self._image_tab_show_tiles[image_id] = not self._image_tab_show_tiles[image_id]
        self._refresh_image_tab(image_id)

    # noinspection PyUnusedLocal
    def _on_image_tab_close_button_click(self, button, image_id):
        self._close_image_tab(image_id)
        self._refresh_image_tab_titles()

    # noinspection PyUnusedLocal
    @traitlets.observe('images')
    def _observe_images(self, change):
        self._update_channel_properties()
        self._update_channel_histograms()
        self._update_gallery_tab_outputs()
        self._refresh_channel_list_box()
        self._refresh_channel_selection_dropdown_options()
        self._refresh_channel_selection_dropdown_value()
        self._refresh_selected_channel_histogram()
        self._refresh_selected_channel_value_range_slider()
        self._refresh_gallery_tab()
        self._close_image_tabs()
        self._refresh_image_tab_titles()
        self._refresh_image_tabs()

    # noinspection PyUnusedLocal
    @traitlets.observe('image_names')
    def _observe_image_names(self, change):
        self._refresh_gallery_tab()
        self._refresh_image_tab_titles()

    # noinspection PyUnusedLocal
    @traitlets.observe('channel_names')
    def _observe_channel_names(self, change):
        self._refresh_channel_list_box()
        self._refresh_channel_selection_dropdown_options()
        self._refresh_channel_selection_dropdown_value()
        self._refresh_image_tabs()

    # noinspection PyUnusedLocal
    @traitlets.observe('channel_vmins')
    @traitlets.observe('channel_vmaxs')
    def _observe_channel_range(self, change):
        self._refresh_selected_channel_value_range_slider()
        self._refresh_gallery_tab()
        self._refresh_image_tabs()

    # noinspection PyUnusedLocal
    @traitlets.observe('channel_states')
    def _observe_channel_states(self, change):
        self._refresh_channel_list_box()
        self._refresh_gallery_tab()
        self._refresh_image_tabs()

    # noinspection PyUnusedLocal
    @traitlets.observe('channel_colors')
    def _observe_channel_colors(self, change):
        self._refresh_channel_list_box()
        self._refresh_gallery_tab()
        self._refresh_image_tabs()

    # noinspection PyUnusedLocal
    @traitlets.observe('num_histogram_bins')
    def _observe_num_histogram_bins(self, change):
        self._update_channel_histograms()
        self._refresh_selected_channel_histogram()

    # noinspection PyUnusedLocal
    @traitlets.observe('selected_channel')
    def _observe_selected_channel(self, change):
        self._refresh_channel_selection_dropdown_value()
        self._refresh_selected_channel_histogram()
        self._refresh_selected_channel_value_range_slider()

    # noinspection PyUnusedLocal
    @traitlets.observe('num_columns')
    def _observe_num_columns(self, change):
        self._refresh_gallery_tab_num_columns()
        self._refresh_image_tabs()


def view_multichannel(images, image_names=None, channel_names=None, channel_vmins=None, channel_vmaxs=None,
                      channel_states=None, channel_colors=None, num_histogram_bins=100, selected_channel=0,
                      num_columns=4, backend='tifffile'):
    assert backend in ['tifffile', 'imageio']
    if not isinstance(images, list):
        images = [images]
    # read images
    if np.all([isinstance(image, str) for image in images]):
        if image_names is None:
            image_names = images
        if backend == 'tifffile':
            from tifffile import imread
            images = [imread(image) for image in images]
        if backend == 'imageio':
            from imageio import imread
            images = [imread(image) for image in images]
    # initialize view
    mv = MultichannelView()
    mv.images = images
    if image_names is not None:
        assert len(image_names) == mv.get_num_images()
        mv.image_names = image_names
    if channel_names is not None:
        assert len(channel_names) == mv.get_num_channels()
        mv.channel_names = channel_names
    if channel_vmins is not None:
        assert len(channel_vmins) == mv.get_num_channels()
        mv.channel_vmins = channel_vmins
    if channel_vmaxs is not None:
        assert len(channel_vmaxs) == mv.get_num_channels()
        mv.channel_vmaxs = channel_vmaxs
    if channel_states is not None:
        assert len(channel_states) == mv.get_num_channels()
        mv.channel_states = channel_states
    if channel_colors is not None:
        assert len(channel_colors) == mv.get_num_channels()
        mv.channel_colors = channel_colors
    mv.num_histogram_bins = num_histogram_bins
    mv.selected_channel = selected_channel
    mv.num_columns = num_columns
    return mv
    # display view
    #from IPython.display import display
    #display((mv,))
